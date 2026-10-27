; Lowkey Media Server — Inno Setup script.
;
; Compiled in CI (release.yml) on the Windows runner AFTER the release
; payload is staged, with:
;   ISCC.exe /DAppVersion=<version> /DStageDir=<abs path to release\windows-amd64> ^
;            /O<output dir> media-server\packaging\windows\setup.iss
;
; Installs per-user (no UAC prompt) to %LocalAppData%\Programs, adds a Start
; Menu entry, an optional start-with-Windows task, and launches the server
; (which opens the browser to the setup wizard) when the user finishes.
;
; The desktop viewer runs this silently with /MERGETASKS="!autostart": the
; viewer starts the server itself (its "Start media server with the viewer"
; setting), so it doesn't also register a login item.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef StageDir
  #define StageDir "..\..\..\release\windows-amd64"
#endif

[Setup]
; Stable AppId so upgrades replace the previous install.
AppId={{8F7C3D2A-91B4-4B6E-A6D0-5A11C10C1D42}
AppName=Lowkey Media Server
AppVersion={#AppVersion}
AppPublisher=Steve Castle
AppPublisherURL=https://github.com/SteveCastle/loki
DefaultDirName={localappdata}\Programs\Lowkey Media Server
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
OutputBaseFilename=lowkey-media-server-setup-windows-amd64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
UninstallDisplayName=Lowkey Media Server
CloseApplications=yes

[Tasks]
Name: "autostart"; Description: "Start Lowkey Media Server when Windows starts (recommended for a server)"; GroupDescription: "Options:"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userprograms}\Lowkey Media Server"; Filename: "{app}\lowkeymediaserver.exe"
; --background: a login start must not pop a browser tab every boot.
Name: "{userstartup}\Lowkey Media Server"; Filename: "{app}\lowkeymediaserver.exe"; Parameters: "--background"; Tasks: autostart

[Run]
Filename: "{app}\lowkeymediaserver.exe"; Description: "Launch Lowkey Media Server now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop the tray/server before removing files.
; /T takes the embed/onnxtag workers down with it.
Filename: "taskkill"; Parameters: "/F /T /IM lowkeymediaserver.exe"; Flags: runhidden; RunOnceId: "KillServer"

[Code]
// An upgrade must replace files a running server holds open, and
// CloseApplications (Restart Manager) can't close a windowless tray app or
// its windowless workers — a silent install would then fail on locked
// files. Stop the server tree first; the desktop viewer (or the user)
// starts the new one afterwards.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /T /IM lowkeymediaserver.exe', '',
    SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := '';
end;
