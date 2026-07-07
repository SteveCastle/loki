# Third-Party Software in Lowkey Media Server

Lowkey Media Server ships with, or invokes, the third-party programs and
libraries below. They are separate works distributed alongside the server
("mere aggregation"); each remains under its own license. This file and the
accompanying license texts satisfy the notice requirements for
redistribution.

To bump a component version, see `media-server/scripts/bundled-versions.json`
— URLs there are pinned so the source links below stay accurate for what was
actually shipped.

---

## FFmpeg (ffmpeg, ffprobe, ffplay) — GPL v3

These builds include GPL-licensed components (notably libx264, used for HLS
transcoding), so the binaries are distributed under the **GNU General Public
License version 3**. The full license text is in `GPL-3.0.txt` next to this
file.

FFmpeg is a trademark of Fabrice Bellard, originator of the FFmpeg project.

Corresponding source code:

- **Windows / Linux builds** (BtbN FFmpeg-Builds, pinned tag
  `autobuild-2026-07-06-14-19`): built from FFmpeg release branch 7.1 at
  commit `7d0e842004` (version `n7.1.5-1-g7d0e842004`) — source at
  <https://github.com/FFmpeg/FFmpeg/commit/7d0e842004> /
  <https://git.ffmpeg.org/ffmpeg.git>; build scripts and dependency
  versions at <https://github.com/BtbN/FFmpeg-Builds> (same tag).
- **macOS builds** (evermeet.cx, version 7.1): sources and build information
  at <https://evermeet.cx/ffmpeg/> and <https://ffmpeg.org/download.html>.
- FFmpeg project source: <https://ffmpeg.org/download.html#get-sources>.

Upon request, the Lowkey Media Server project will provide a copy of the
corresponding source for the FFmpeg binaries shipped with a given release.

## ExifTool — Perl Artistic License / GPL (dual-licensed)

ExifTool by Phil Harvey, distributed under the same terms as Perl itself:
your choice of the GNU General Public License (see `GPL-3.0.txt`) or the
Perl Artistic License (<https://dev.perl.org/licenses/artistic.html>).

- Project: <https://exiftool.org/>
- Source: <https://sourceforge.net/projects/exiftool/files/>
- The Windows package additionally bundles a minimal Perl runtime
  (Strawberry Perl components) under the same Perl licensing terms.

## ONNX Runtime — MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

- Project and source: <https://github.com/microsoft/onnxruntime>
  (pinned release: v1.22.0)

## AI models — downloaded by the user, not redistributed

The optional AI models (auto-tagging, visual search, face recognition,
transcription) are **not included in this distribution**. The setup wizard
and the Dependencies page download them directly from their upstream
publishers at the user's request, and each model remains under its
publisher's license (Apache-2.0, MIT, or OpenRAIL depending on the model —
shown in the Dependencies UI). Consult the model pages linked there before
commercial use; OpenRAIL-licensed models (e.g. CCIP) carry use restrictions.

## Optional tools — detected, never distributed

yt-dlp, gallery-dl, ollama, and DiscordChatExporter are detected on your
system PATH if you installed them yourself; they are not part of this
distribution.
