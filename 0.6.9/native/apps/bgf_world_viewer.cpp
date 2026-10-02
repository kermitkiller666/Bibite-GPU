#include "bgf_world.h"

#include <windows.h>
#include <commctrl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr wchar_t kWindowClass[] = L"BibitesGpuWorldWindow";
constexpr UINT kSnapshotReady = WM_APP + 1;
constexpr UINT kAutoStart = WM_APP + 2;
constexpr int kGpuCombo = 1001;
constexpr int kSpeedCombo = 1002;
constexpr int kRestartButton = 1003;
constexpr int kPauseButton = 1004;
constexpr int kSimulationTop = 72;
constexpr int kSpeedValues[] = {
    1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 0};

struct AppState {
    HWND window = nullptr;
    HWND gpu_combo = nullptr;
    HWND speed_combo = nullptr;
    HWND pause_button = nullptr;
    std::thread worker;
    std::atomic<bool> stop{false};
    std::atomic<bool> paused{false};
    std::atomic<int> speed_index{14};
    std::mutex snapshot_mutex;
    std::vector<BgfWorldBibite> snapshot;
    BgfWorldStats stats{};
    double gpu_milliseconds = 0.0;
    double wall_milliseconds = 0.0;
    double offload_percent = 0.0;
    double engine_multiplier = 0.0;
    std::wstring device_name;
    std::wstring status = L"Starting GPU-resident world...";
    HBRUSH palette[8]{};
};

std::wstring widen(const char* text)
{
    if (!text || !*text) return {};
    const int length = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (length <= 1) return {};
    std::wstring result(static_cast<size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, result.data(), length);
    result.resize(static_cast<size_t>(length - 1));
    return result;
}

void set_status(AppState* state, const std::wstring& value)
{
    std::lock_guard<std::mutex> lock(state->snapshot_mutex);
    state->status = value;
}

void simulation_worker(AppState* state, int32_t device_index)
{
    BgfWorldConfig config{};
    bgf_world_default_config(&config);
    config.device_index = device_index;
    config.max_bibites = 2048;
    config.initial_bibites = 512;
    config.pellet_count = 8192;

    BgfWorldHandle world = nullptr;
    const int create_status = bgf_world_create(&config, &world);
    if (create_status != 0) {
        std::wostringstream message;
        message << L"Could not create CUDA world (error " << create_status << L").";
        set_status(state, message.str());
        PostMessageW(state->window, kSnapshotReady, 0, 0);
        return;
    }

    char device_name[256]{};
    bgf_world_get_device_name(world, device_name, sizeof(device_name));
    {
        std::lock_guard<std::mutex> lock(state->snapshot_mutex);
        state->device_name = widen(device_name);
        state->status = L"Simulation state is resident on the GPU.";
        state->gpu_milliseconds = 0.0;
        state->wall_milliseconds = 0.0;
        state->offload_percent = 0.0;
        state->engine_multiplier = 0.0;
        state->snapshot.clear();
        state->stats = {};
    }

    double total_gpu_ms = 0.0;
    double total_wall_ms = 0.0;
    std::vector<BgfWorldBibite> next_snapshot(static_cast<size_t>(config.max_bibites));
    std::ofstream history("overnight-history.csv", std::ios::binary | std::ios::trunc);
    if (history) {
        history << "local_time,completed_steps,simulated_days,living,births,deaths,"
                   "pellets_eaten,gpu_offload_percent,engine_multiplier\n";
        history.flush();
    }
    auto next_report = std::chrono::steady_clock::now();
    while (!state->stop.load(std::memory_order_relaxed)) {
        if (state->paused.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            continue;
        }

        const int selected = std::clamp(
            state->speed_index.load(std::memory_order_relaxed),
            0,
            static_cast<int>(std::size(kSpeedValues)) - 1);
        const int multiplier = kSpeedValues[selected];
        const int32_t steps = multiplier == 0
            ? 128
            : std::clamp((multiplier + 7) / 8, 1, 128);
        const auto batch_started = std::chrono::steady_clock::now();
        BgfWorldStepMetrics metrics{};
        const int step_status = bgf_world_step(world, steps, &metrics);
        if (step_status != 0) {
            std::wostringstream message;
            message << L"GPU simulation failed (CUDA error " << step_status << L").";
            set_status(state, message.str());
            break;
        }
        total_gpu_ms += metrics.gpu_milliseconds;
        total_wall_ms += metrics.wall_milliseconds;

        BgfWorldStats stats{};
        int32_t written = 0;
        const int stats_status = bgf_world_get_stats(world, &stats);
        const int snapshot_status = stats_status == 0
            ? bgf_world_download_bibites(
                world,
                next_snapshot.data(),
                static_cast<int32_t>(next_snapshot.size()),
                &written)
            : stats_status;
        if (snapshot_status != 0) {
            std::wostringstream message;
            message << L"Snapshot download failed (CUDA error " << snapshot_status << L").";
            set_status(state, message.str());
            break;
        }

        const double host_ms = std::max(total_wall_ms - total_gpu_ms, 0.0);
        const double measured_ms = total_gpu_ms + host_ms;
        const double offload_percent = measured_ms > 0.0
            ? total_gpu_ms / measured_ms * 100.0 : 0.0;
        const double engine_multiplier = total_wall_ms > 0.0
            ? stats.simulated_seconds / (total_wall_ms / 1000.0) : 0.0;
        {
            std::lock_guard<std::mutex> lock(state->snapshot_mutex);
            state->snapshot.assign(next_snapshot.begin(), next_snapshot.begin() + written);
            state->stats = stats;
            state->gpu_milliseconds = total_gpu_ms;
            state->wall_milliseconds = total_wall_ms;
            state->offload_percent = offload_percent;
            state->engine_multiplier = engine_multiplier;
        }
        PostMessageW(state->window, kSnapshotReady, 0, 0);

        const auto report_time = std::chrono::steady_clock::now();
        if (report_time >= next_report) {
            SYSTEMTIME local_time{};
            GetLocalTime(&local_time);
            std::ofstream report("overnight-status.json", std::ios::binary | std::ios::trunc);
            if (report) {
                report << std::fixed << std::setprecision(6)
                       << "{\n"
                       << "  \"status\": \"running\",\n"
                       << "  \"device_index\": " << device_index << ",\n"
                       << "  \"device_name\": \"" << device_name << "\",\n"
                       << "  \"initial_bibites\": " << config.initial_bibites << ",\n"
                       << "  \"population_cap\": " << config.max_bibites << ",\n"
                       << "  \"pellets\": " << config.pellet_count << ",\n"
                       << "  \"completed_steps\": " << stats.completed_steps << ",\n"
                       << "  \"simulated_days\": " << stats.simulated_seconds / 86400.0 << ",\n"
                       << "  \"living_bibites\": " << stats.living_bibites << ",\n"
                       << "  \"births\": " << stats.births << ",\n"
                       << "  \"deaths\": " << stats.deaths << ",\n"
                       << "  \"pellets_eaten\": " << stats.pellets_eaten << ",\n"
                       << "  \"gpu_offload_percent\": " << offload_percent << ",\n"
                       << "  \"engine_multiplier\": " << engine_multiplier << "\n"
                       << "}\n";
            }
            if (history) {
                history << std::setfill('0')
                        << local_time.wYear << '-'
                        << std::setw(2) << local_time.wMonth << '-'
                        << std::setw(2) << local_time.wDay << ' '
                        << std::setw(2) << local_time.wHour << ':'
                        << std::setw(2) << local_time.wMinute << ':'
                        << std::setw(2) << local_time.wSecond << ','
                        << stats.completed_steps << ','
                        << std::fixed << std::setprecision(6)
                        << stats.simulated_seconds / 86400.0 << ','
                        << stats.living_bibites << ','
                        << stats.births << ','
                        << stats.deaths << ','
                        << stats.pellets_eaten << ','
                        << offload_percent << ','
                        << engine_multiplier << '\n';
                history.flush();
            }
            next_report = report_time + std::chrono::seconds(30);
        }

        if (multiplier > 0) {
            const double target_seconds =
                static_cast<double>(steps) * config.fixed_delta_time / multiplier;
            const double elapsed_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - batch_started).count();
            if (target_seconds > elapsed_seconds) {
                std::this_thread::sleep_for(std::chrono::duration<double>(
                    target_seconds - elapsed_seconds));
            }
        }
    }

    bgf_world_destroy(world);
    PostMessageW(state->window, kSnapshotReady, 0, 0);
}

void stop_worker(AppState* state)
{
    state->stop.store(true, std::memory_order_relaxed);
    if (state->worker.joinable()) state->worker.join();
}

void start_worker(AppState* state)
{
    stop_worker(state);
    state->stop.store(false, std::memory_order_relaxed);
    state->paused.store(false, std::memory_order_relaxed);
    SetWindowTextW(state->pause_button, L"Pause");
    const LRESULT selected = SendMessageW(state->gpu_combo, CB_GETCURSEL, 0, 0);
    const int32_t device = selected == CB_ERR ? 0 : static_cast<int32_t>(selected);
    set_status(state, L"Creating persistent GPU world...");
    state->worker = std::thread(simulation_worker, state, device);
}

void paint_world(HWND window, AppState* state)
{
    PAINTSTRUCT paint{};
    HDC target = BeginPaint(window, &paint);
    RECT client{};
    GetClientRect(window, &client);
    const int width = std::max(client.right - client.left, 1L);
    const int height = std::max(client.bottom - client.top, 1L);
    HDC buffer = CreateCompatibleDC(target);
    HBITMAP bitmap = CreateCompatibleBitmap(target, width, height);
    HGDIOBJ old_bitmap = SelectObject(buffer, bitmap);
    HBRUSH background = CreateSolidBrush(RGB(8, 10, 14));
    FillRect(buffer, &client, background);
    DeleteObject(background);

    std::vector<BgfWorldBibite> snapshot;
    BgfWorldStats stats{};
    double offload = 0.0;
    double multiplier = 0.0;
    std::wstring device_name;
    std::wstring status;
    {
        std::lock_guard<std::mutex> lock(state->snapshot_mutex);
        snapshot = state->snapshot;
        stats = state->stats;
        offload = state->offload_percent;
        multiplier = state->engine_multiplier;
        device_name = state->device_name;
        status = state->status;
    }

    const int simulation_height = std::max(height - kSimulationTop, 1);
    const float scale_x = static_cast<float>(width) / 1000.0f;
    const float scale_y = static_cast<float>(simulation_height) / 1000.0f;
    SetBkMode(buffer, TRANSPARENT);
    for (const BgfWorldBibite& bibite : snapshot) {
        const int x = static_cast<int>((bibite.position_x + 500.0f) * scale_x);
        const int y = kSimulationTop + simulation_height -
            static_cast<int>((bibite.position_y + 500.0f) * scale_y);
        const int palette_index =
            (bibite.color_r > 0.5f ? 1 : 0) |
            (bibite.color_g > 0.5f ? 2 : 0) |
            (bibite.color_b > 0.5f ? 4 : 0);
        HGDIOBJ old_brush = SelectObject(buffer, state->palette[palette_index]);
        HGDIOBJ old_pen = SelectObject(buffer, GetStockObject(NULL_PEN));
        const int radius = std::clamp(static_cast<int>(bibite.size * 1.8f), 1, 4);
        Ellipse(buffer, x - radius, y - radius, x + radius + 1, y + radius + 1);
        SelectObject(buffer, old_pen);
        SelectObject(buffer, old_brush);
    }

    SetTextColor(buffer, RGB(220, 235, 245));
    std::wostringstream first_line;
    first_line.setf(std::ios::fixed);
    first_line.precision(3);
    first_line << (device_name.empty() ? L"CUDA device starting" : device_name)
               << L"   |   GPU simulation share " << offload << L"%"
               << L"   |   engine " << multiplier << L"x";
    TextOutW(buffer, 12, 48, first_line.str().c_str(),
        static_cast<int>(first_line.str().size()));

    std::wostringstream second_line;
    second_line.setf(std::ios::fixed);
    second_line.precision(2);
    second_line << L"Step " << stats.completed_steps
                << L"   living " << stats.living_bibites
                << L"   births " << stats.births
                << L"   deaths " << stats.deaths
                << L"   eaten " << stats.pellets_eaten
                << L"   simulated " << stats.simulated_seconds / 86400.0 << L" days";
    TextOutW(buffer, 12, height - 24, second_line.str().c_str(),
        static_cast<int>(second_line.str().size()));
    SetTextColor(buffer, RGB(130, 165, 185));
    TextOutW(buffer, std::max(width - 360, 12), 48, status.c_str(),
        static_cast<int>(status.size()));

    BitBlt(target, 0, 0, width, height, buffer, 0, 0, SRCCOPY);
    SelectObject(buffer, old_bitmap);
    DeleteObject(bitmap);
    DeleteDC(buffer);
    EndPaint(window, &paint);
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w_param, LPARAM l_param)
{
    AppState* state = reinterpret_cast<AppState*>(GetWindowLongPtrW(window, GWLP_USERDATA));
    switch (message) {
        case WM_CREATE: {
            state = new AppState{};
            state->window = window;
            SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
            const COLORREF colors[8] = {
                RGB(150, 150, 150), RGB(235, 90, 90), RGB(90, 235, 110), RGB(235, 220, 90),
                RGB(90, 130, 235), RGB(220, 90, 235), RGB(90, 230, 230), RGB(240, 240, 240)};
            for (int index = 0; index < 8; ++index) state->palette[index] = CreateSolidBrush(colors[index]);

            CreateWindowExW(0, L"STATIC", L"GPU:", WS_CHILD | WS_VISIBLE,
                12, 13, 35, 22, window, nullptr, nullptr, nullptr);
            state->gpu_combo = CreateWindowExW(0, WC_COMBOBOXW, nullptr,
                WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
                48, 9, 260, 300, window,
                reinterpret_cast<HMENU>(static_cast<INT_PTR>(kGpuCombo)), nullptr, nullptr);
            int32_t device_count = 0;
            if (bgf_get_device_count(&device_count) == 0) {
                for (int32_t device = 0; device < device_count; ++device) {
                    char name[256]{};
                    if (bgf_get_device_name(device, name, sizeof(name)) == 0) {
                        const std::wstring label = std::to_wstring(device) + L": " + widen(name);
                        SendMessageW(state->gpu_combo, CB_ADDSTRING, 0,
                            reinterpret_cast<LPARAM>(label.c_str()));
                    }
                }
            }
            SendMessageW(state->gpu_combo, CB_SETCURSEL, 0, 0);

            CreateWindowExW(0, L"STATIC", L"Speed:", WS_CHILD | WS_VISIBLE,
                320, 13, 45, 22, window, nullptr, nullptr, nullptr);
            state->speed_combo = CreateWindowExW(0, WC_COMBOBOXW, nullptr,
                WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                368, 9, 110, 300, window,
                reinterpret_cast<HMENU>(static_cast<INT_PTR>(kSpeedCombo)), nullptr, nullptr);
            for (int value : kSpeedValues) {
                const std::wstring label = value == 0
                    ? L"MAX GPU" : std::to_wstring(value) + L"x";
                SendMessageW(state->speed_combo, CB_ADDSTRING, 0,
                    reinterpret_cast<LPARAM>(label.c_str()));
            }
            SendMessageW(state->speed_combo, CB_SETCURSEL, 11, 0);

            CreateWindowExW(0, WC_BUTTONW, L"Restart",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                490, 8, 90, 28, window,
                reinterpret_cast<HMENU>(static_cast<INT_PTR>(kRestartButton)), nullptr, nullptr);
            state->pause_button = CreateWindowExW(0, WC_BUTTONW, L"Pause",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                590, 8, 90, 28, window,
                reinterpret_cast<HMENU>(static_cast<INT_PTR>(kPauseButton)), nullptr, nullptr);
            PostMessageW(window, kAutoStart, 0, 0);
            return 0;
        }
        case kAutoStart:
            if (state) start_worker(state);
            return 0;
        case kSnapshotReady:
            InvalidateRect(window, nullptr, FALSE);
            return 0;
        case WM_COMMAND:
            if (!state) break;
            if (LOWORD(w_param) == kRestartButton && HIWORD(w_param) == BN_CLICKED) {
                start_worker(state);
                return 0;
            }
            if (LOWORD(w_param) == kPauseButton && HIWORD(w_param) == BN_CLICKED) {
                const bool pause = !state->paused.load(std::memory_order_relaxed);
                state->paused.store(pause, std::memory_order_relaxed);
                SetWindowTextW(state->pause_button, pause ? L"Resume" : L"Pause");
                return 0;
            }
            if (LOWORD(w_param) == kSpeedCombo && HIWORD(w_param) == CBN_SELCHANGE) {
                const LRESULT selected = SendMessageW(state->speed_combo, CB_GETCURSEL, 0, 0);
                if (selected != CB_ERR) state->speed_index.store(static_cast<int>(selected));
                return 0;
            }
            break;
        case WM_PAINT:
            if (state) paint_world(window, state);
            else DefWindowProcW(window, message, w_param, l_param);
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case WM_SIZE:
            InvalidateRect(window, nullptr, FALSE);
            return 0;
        case WM_DESTROY:
            if (state) {
                stop_worker(state);
                for (HBRUSH brush : state->palette) DeleteObject(brush);
                delete state;
                SetWindowLongPtrW(window, GWLP_USERDATA, 0);
            }
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(window, message, w_param, l_param);
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show_command)
{
    INITCOMMONCONTROLSEX controls{sizeof(controls), ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&controls);
    WNDCLASSEXW window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.style = CS_HREDRAW | CS_VREDRAW;
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = instance;
    window_class.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    window_class.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    window_class.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    window_class.lpszClassName = kWindowClass;
    if (!RegisterClassExW(&window_class)) return 1;

    HWND window = CreateWindowExW(
        0,
        kWindowClass,
        L"Bibites GPU World - 99% GPU-resident evolution engine",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1280,
        820,
        nullptr,
        nullptr,
        instance,
        nullptr);
    if (!window) return 2;
    ShowWindow(window, show_command);
    UpdateWindow(window);

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return static_cast<int>(message.wParam);
}
