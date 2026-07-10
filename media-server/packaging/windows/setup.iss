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
Name: "{userstartup}\Lowkey Media Server"; Filename: "{app}\lowkeymediaserver.exe"; Tasks: autostart

[Run]
Filename: "{app}\lowkeymediaserver.exe"; Description: "Launch Lowkey Media Server now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop the tray/server before removing files.
Filename: "taskkill"; Parameters: "/F /IM lowkeymediaserver.exe"; Flags: runhidden; RunOnceId: "KillServer"
