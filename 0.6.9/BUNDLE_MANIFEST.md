# Portable 0.6.9 bundle manifest

The portable EXE is an unofficial, single-download Windows x64 package. It
contains the publicly released **The Bibites 0.6.3.1** game from
[the creators' itch.io page](https://thebibites.itch.io/the-bibites), the
0.6.9 GPU plugin, BepInEx 5.4.23.5, and UnityDoorstop 4.5.0. It does not
contain a Patreon-only build, pre-existing save, personal configuration,
benchmark result, screenshot, or prior runtime log.

| File | SHA-256 |
| --- | --- |
| `Bibites GPU Fork 0.6.9 - PORTABLE.exe` | `80E14864567F6ED48382A50F5A162233800F3EA19B831977BE9676F557B55BCC` |
| Original `The Bibites 0.6.3.1 - Windows 64x.zip` input | `60879A1964D55FDF176052817B0356AED02FF601F4A8A1A70CFE2427C72FEFB1` |
| Embedded ZIP payload | `60029381B082B8B055EF60B2E81CD56FB19962CBE90361E6C8CD0FC36F4F3B47` |
| `BibitesGpuFork.dll` | `2250B4C2F6345E6539C23B3FB0A5215918CFA751445A963565E23C427895A971` |
| `BibitesGpuFork.Core.dll` | `BD9628C97089E4EAB943E7689193145E66167F1EF687857C87B88F06C59F154A` |
| `BibitesGpuNative.dll` | `911CEDC197FF4C855EE2698EE2DEAD9597157C944153304CCDE64179874ADD9E` |
| UnityDoorstop `winhttp.dll` | `8C6CDBC38836DEE87E3368F5DE1994D7C0CCEBF29E4CE7ABA3C0981F9375412C` |

The EXE is **65,822,208 bytes**. Its embedded ZIP has 243 files and expands
to approximately 161 MiB. On first launch the program verifies the embedded
ZIP hash, extracts it into `Bibites GPU Fork 0.6.9 - data-60029381b082`
beside the EXE, and opens `The Bibites.exe`. The folder is reused on later
launches; it is intentionally not deleted after exit. BepInEx and
UnityDoorstop license texts are included inside that folder under `LICENSES`.

A bounded local smoke check reached the title screen and started a default
512-starter / 2,048-cap GPU world using the RTX 4070 Ti. The BepInEx log
reported that the 0.6.9 plugin loaded. This is **not** a long-run stability,
feeding-animation, save/reload, or performance certification.
