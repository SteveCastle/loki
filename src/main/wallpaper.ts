// Per-monitor desktop wallpaper (Windows). Uses the shell's IDesktopWallpaper
// COM interface through a short PowerShell/C# shim, since Electron has no
// wallpaper API and the classic SystemParametersInfo call can only set every
// monitor at once.
import { app, ipcMain, nativeImage } from 'electron';
import { execFile } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

export interface MonitorInfo {
  /** IDesktopWallpaper device path — the id to pass back to set-wallpaper. */
  id: string;
  index: number;
  left: number;
  top: number;
  width: number;
  height: number;
  primary: boolean;
}

const SCRIPT = `
param([string]$Mode, [string]$MonitorId, [string]$ImagePath)
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
[StructLayout(LayoutKind.Sequential)]
public struct WpRect { public int Left, Top, Right, Bottom; }
[ComImport, Guid("B92B56A9-8B55-4E14-9A89-0199BBB6F93B"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IDesktopWallpaper {
  void SetWallpaper([MarshalAs(UnmanagedType.LPWStr)] string monitorID,
                    [MarshalAs(UnmanagedType.LPWStr)] string wallpaper);
  [return: MarshalAs(UnmanagedType.LPWStr)]
  string GetWallpaper([MarshalAs(UnmanagedType.LPWStr)] string monitorID);
  [return: MarshalAs(UnmanagedType.LPWStr)]
  string GetMonitorDevicePathAt(uint monitorIndex);
  uint GetMonitorDevicePathCount();
  WpRect GetMonitorRECT([MarshalAs(UnmanagedType.LPWStr)] string monitorID);
  void SetBackgroundColor(uint color);
  uint GetBackgroundColor();
  void SetPosition(int position);
}
[ComImport, Guid("C2CF3110-460E-4fc1-B9D0-8A1C0C9CC4BD")]
public class DesktopWallpaperClass { }
public static class Wp {
  static IDesktopWallpaper Dw() {
    return (IDesktopWallpaper)new DesktopWallpaperClass();
  }
  // One monitor per line: id|index|left|top|width|height
  public static string List() {
    var dw = Dw();
    var sb = new System.Text.StringBuilder();
    uint n = dw.GetMonitorDevicePathCount();
    for (uint i = 0; i < n; i++) {
      string id = dw.GetMonitorDevicePathAt(i);
      if (string.IsNullOrEmpty(id)) continue;
      WpRect r;
      // Known monitors that are not currently attached fail here; skip them.
      try { r = dw.GetMonitorRECT(id); } catch (COMException) { continue; }
      sb.Append(id + "|" + i + "|" + r.Left + "|" + r.Top + "|"
        + (r.Right - r.Left) + "|" + (r.Bottom - r.Top) + "\\n");
    }
    return sb.ToString();
  }
  public static void Set(string monitorId, string image) {
    var dw = Dw();
    dw.SetPosition(4); // DWPOS_FILL
    dw.SetWallpaper(monitorId == "*" ? null : monitorId, image);
  }
}
"@
if ($Mode -eq "list") { [Wp]::List() }
elseif ($Mode -eq "set") { [Wp]::Set($MonitorId, $ImagePath) }
`;

let scriptPath: string | null = null;

function ensureScript(): string {
  if (scriptPath && fs.existsSync(scriptPath)) return scriptPath;
  const dir = path.join(app.getPath('userData'), 'wallpaper');
  fs.mkdirSync(dir, { recursive: true });
  scriptPath = path.join(dir, 'wallpaper.ps1');
  fs.writeFileSync(scriptPath, SCRIPT, 'utf8');
  return scriptPath;
}

function runScript(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        ensureScript(),
        ...args,
      ],
      { windowsHide: true, timeout: 15000 },
      (err, stdout, stderr) => {
        if (err) reject(new Error((stderr || err.message).trim()));
        else resolve(stdout.trim());
      }
    );
  });
}

// Windows wallpapers must be a format the shell decodes; anything else
// (webp, avif, ...) is converted once to a stable PNG under userData.
const NATIVE_EXT = new Set(['.jpg', '.jpeg', '.jfif', '.png', '.bmp']);

function wallpaperFile(source: string): string {
  if (NATIVE_EXT.has(path.extname(source).toLowerCase())) return source;
  const img = nativeImage.createFromPath(source);
  if (img.isEmpty()) {
    throw new Error('This image format can’t be used as a wallpaper');
  }
  const stat = fs.statSync(source);
  const key = crypto
    .createHash('sha1')
    .update(`${source}|${stat.mtimeMs}|${stat.size}`)
    .digest('hex')
    .slice(0, 16);
  const dir = path.join(app.getPath('userData'), 'wallpaper');
  fs.mkdirSync(dir, { recursive: true });
  const out = path.join(dir, `${key}.png`);
  if (!fs.existsSync(out)) fs.writeFileSync(out, img.toPNG());
  return out;
}

export function registerWallpaperHandlers() {
  ipcMain.handle('list-monitors', async (): Promise<MonitorInfo[]> => {
    if (process.platform !== 'win32') return [];
    const raw = await runScript(['list']);
    return raw
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => {
        const [id, index, left, top, width, height] = line.split('|');
        return {
          id,
          index: Number(index),
          left: Number(left),
          top: Number(top),
          width: Number(width),
          height: Number(height),
          primary: Number(left) === 0 && Number(top) === 0,
        };
      });
  });

  // args: [imagePath, monitorId] — monitorId '*' sets every monitor.
  ipcMain.handle('set-wallpaper', async (_event, args: unknown[]) => {
    if (process.platform !== 'win32') {
      throw new Error('Setting a wallpaper is only supported on Windows');
    }
    const [source, monitorId] = args as [string, string];
    if (typeof source !== 'string' || !fs.existsSync(source)) {
      throw new Error('Image file not found');
    }
    if (typeof monitorId !== 'string' || !monitorId) {
      throw new Error('No monitor specified');
    }
    await runScript(['set', monitorId, wallpaperFile(source)]);
    return true;
  });
}
