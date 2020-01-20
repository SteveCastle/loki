const { app, Menu, shell, dialog } = require("electron");
const btoa = require("btoa");
const path = require("path");
const isDev = require("electron-is-dev");

class MenuBuilder {
  mainWindow;

  constructor(mainWindow) {
    this.mainWindow = mainWindow;
  }

  buildMenu() {
    if (
      process.env.NODE_ENV === "development" ||
      process.env.DEBUG_PROD === "true"
    ) {
      this.setupDevelopmentEnvironment();
    }

    let template;

    if (process.platform === "darwin") {
      template = this.buildDarwinTemplate();
    } else {
      template = this.buildDefaultTemplate();
    }

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);

    return menu;
  }

  setupDevelopmentEnvironment() {
    this.mainWindow.openDevTools();
    this.mainWindow.webContents.on("context-menu", (e, props) => {
      const { x, y } = props;

      Menu.buildFromTemplate([
        {
          label: "Inspect element",
          click: () => {
            this.mainWindow.inspectElement(x, y);
          }
        }
      ]).popup(this.mainWindow);
    });
  }

  buildDarwinTemplate() {
    const subMenuAbout = {
      label: "Low Key Image Viewer",
      submenu: [
        {
          label: "About LowKey Image Viewer",
          selector: "orderFrontStandardAboutPanel:"
        },
        { type: "separator" },
        {
          label: "Quit",
          accelerator: "Command+Q",
          click: () => {
            app.quit();
          }
        }
      ]
    };
    const subMenuFile = {
      label: "File",
      submenu: [
        {
          label: "Open",
          accelerator: "Command+M",
          click: () => {
            dialog
              .showOpenDialog(this.mainWindow, {
                properties: ["openFile"]
              })
              .then(result => {
                !result.canceled &&
                  this.mainWindow.loadURL(
                    isDev
                      ? `http://localhost:3000?${btoa(result.filePaths[0])}`
                      : `file://${path.join(
                          __dirname,
                          `../build/index.html?${btoa(result.filePaths[0])}`
                        )}`
                  );
              })
              .catch(err => {
                console.log(err);
              });
          }
        }
      ]
    };
    const subMenuViewDev = {
      label: "View",
      submenu: [
        {
          label: "Reload",
          accelerator: "Command+R",
          click: () => {
            this.mainWindow.webContents.reload();
          }
        },
        {
          label: "Toggle Full Screen",
          accelerator: "Ctrl+Command+F",
          click: () => {
            this.mainWindow.setFullScreen(!this.mainWindow.isFullScreen());
          }
        },
        {
          label: "Toggle Developer Tools",
          accelerator: "Alt+Command+I",
          click: () => {
            this.mainWindow.toggleDevTools();
          }
        }
      ]
    };
    const subMenuViewProd = {
      label: "View",
      submenu: [
        {
          label: "Toggle Full Screen",
          accelerator: "Ctrl+Command+F",
          click: () => {
            this.mainWindow.setFullScreen(!this.mainWindow.isFullScreen());
          }
        }
      ]
    };
    const subMenuWindow = {
      label: "Window",
      submenu: [
        {
          label: "Minimize",
          accelerator: "Command+M",
          selector: "performMiniaturize:"
        },
        { label: "Close", accelerator: "Command+W", selector: "performClose:" },
        { type: "separator" },
        { label: "Bring All to Front", selector: "arrangeInFront:" }
      ]
    };
    const subMenuHelp = {
      label: "Help",
      submenu: [
        {
          label: "Learn More",
          click() {
            shell.openExternal("https://lowkeyviewer.com");
          }
        },
        {
          label: "Documentation",
          click() {
            shell.openExternal("https://lowkeyviewer.com/docs");
          }
        },
        {
          label: "Community Discussions",
          click() {
            shell.openExternal("https://lowkeyviewer.com/community");
          }
        },
        {
          label: "Search Issues",
          click() {
            shell.openExternal("https://lowkeyviewer.com/community/issues");
          }
        }
      ]
    };

    const subMenuView = subMenuViewDev;

    return [subMenuAbout, subMenuFile, subMenuView, subMenuWindow, subMenuHelp];
  }

  buildDefaultTemplate() {
    const templateDefault = [
      {
        label: "&File",
        submenu: [
          {
            label: "&Open",
            accelerator: "Ctrl+O"
          },
          {
            label: "&Close",
            accelerator: "Ctrl+W",
            click: () => {
              this.mainWindow.close();
            }
          }
        ]
      },
      {
        label: "&View",
        submenu:
          process.env.NODE_ENV === "development"
            ? [
                {
                  label: "&Reload",
                  accelerator: "Ctrl+R",
                  click: () => {
                    this.mainWindow.webContents.reload();
                  }
                },
                {
                  label: "Toggle &Full Screen",
                  accelerator: "F11",
                  click: () => {
                    this.mainWindow.setFullScreen(
                      !this.mainWindow.isFullScreen()
                    );
                  }
                },
                {
                  label: "Toggle &Developer Tools",
                  accelerator: "Alt+Ctrl+I",
                  click: () => {
                    this.mainWindow.toggleDevTools();
                  }
                }
              ]
            : [
                {
                  label: "Toggle &Full Screen",
                  accelerator: "F11",
                  click: () => {
                    this.mainWindow.setFullScreen(
                      !this.mainWindow.isFullScreen()
                    );
                  }
                }
              ]
      },
      {
        label: "Help",
        submenu: [
          {
            label: "Learn More",
            click() {
              shell.openExternal("http://electron.atom.io");
            }
          },
          {
            label: "Documentation",
            click() {
              shell.openExternal(
                "https://github.com/atom/electron/tree/master/docs#readme"
              );
            }
          },
          {
            label: "Community Discussions",
            click() {
              shell.openExternal("https://discuss.atom.io/c/electron");
            }
          },
          {
            label: "Search Issues",
            click() {
              shell.openExternal("https://github.com/atom/electron/issues");
            }
          }
        ]
      }
    ];

    return templateDefault;
  }
}

exports.MenuBuilder = MenuBuilder;
