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
          },
        },
      ]).popup(this.mainWindow);
    });
  }

  buildDarwinTemplate() {
    const subMenuAbout = {
      label: "Low Key Image Viewer",
      submenu: [
        { type: "separator" },
        {
          label: "Quit",
          accelerator: "Command+Q",
          click: () => {
            app.quit();
          },
        },
      ],
    };
    const subMenuFile = {
      label: "File",
      submenu: [],
    };
    const subMenuViewDev = {
      label: "View",
      submenu: [
        {
          label: "Toggle Full Screen",
          accelerator: "Ctrl+Command+F",
          click: () => {
            this.mainWindow.setFullScreen(!this.mainWindow.isFullScreen());
          },
        },
        {
          label: "Open Dev Tools",
          accelerator: "Ctrl+Command+I",
          click: () => {
            this.mainWindow.openDevTools();
          },
        },
        {
          label: "Toggle Always On Top",
          accelerator: "Ctrl+Command+T",
          click: () => {
            this.mainWindow.setAlwaysOnTop(!this.mainWindow.isAlwaysOnTop());
          },
        },
      ],
    };
    const subMenuEdit = {
      label: "Edit",
      submenu: [
        { label: "Undo", accelerator: "CmdOrCtrl+Z", selector: "undo:" },
        { label: "Redo", accelerator: "Shift+CmdOrCtrl+Z", selector: "redo:" },
        { type: "separator" },
        { label: "Cut", accelerator: "CmdOrCtrl+X", selector: "cut:" },
        { label: "Copy", accelerator: "CmdOrCtrl+C", selector: "copy:" },
        { label: "Paste", accelerator: "CmdOrCtrl+V", selector: "paste:" },
        {
          label: "Select All",
          accelerator: "CmdOrCtrl+A",
          selector: "selectAll:",
        },
      ],
    };
    const subMenuHelp = {
      label: "Help",
      submenu: [
        {
          label: "Help",
          click() {
            shell.openExternal("https://lowkeyviewer.com/#features");
          },
        },
        {
          label: "Support",
          click() {
            shell.openExternal("https://lowkeyviewer.com/#support");
          },
        },
      ],
    };

    const subMenuView = subMenuViewDev;

    return [subMenuAbout, subMenuView, subMenuEdit, subMenuHelp];
  }

  buildDefaultTemplate() {
    const templateDefault = [
      {
        label: "&File",
        submenu: [
          {
            label: "&Open",
            accelerator: "Ctrl+O",
          },
          {
            label: "&Close",
            accelerator: "Ctrl+W",
            click: () => {
              this.mainWindow.close();
            },
          },
        ],
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
                  },
                },
                {
                  label: "Toggle &Full Screen",
                  accelerator: "F11",
                  click: () => {
                    this.mainWindow.setFullScreen(
                      !this.mainWindow.isFullScreen()
                    );
                  },
                },
                {
                  label: "Toggle &Developer Tools",
                  accelerator: "Alt+Ctrl+I",
                  click: () => {
                    this.mainWindow.toggleDevTools();
                  },
                },
              ]
            : [
                {
                  label: "Toggle &Full Screen",
                  accelerator: "F11",
                  click: () => {
                    this.mainWindow.setFullScreen(
                      !this.mainWindow.isFullScreen()
                    );
                  },
                },
              ],
      },
      {
        label: "Edit",
        submenu: [
          { label: "Undo", accelerator: "CmdOrCtrl+Z", selector: "undo:" },
          {
            label: "Redo",
            accelerator: "Shift+CmdOrCtrl+Z",
            selector: "redo:",
          },
          { type: "separator" },
          { label: "Cut", accelerator: "CmdOrCtrl+X", selector: "cut:" },
          { label: "Copy", accelerator: "CmdOrCtrl+C", selector: "copy:" },
          { label: "Paste", accelerator: "CmdOrCtrl+V", selector: "paste:" },
          {
            label: "Select All",
            accelerator: "CmdOrCtrl+A",
            selector: "selectAll:",
          },
        ],
      },
      {
        label: "Help",
        submenu: [
          {
            label: "Learn More",
            click() {
              shell.openExternal("http://electron.atom.io");
            },
          },
          {
            label: "Documentation",
            click() {
              shell.openExternal(
                "https://github.com/atom/electron/tree/master/docs#readme"
              );
            },
          },
          {
            label: "Community Discussions",
            click() {
              shell.openExternal("https://discuss.atom.io/c/electron");
            },
          },
          {
            label: "Search Issues",
            click() {
              shell.openExternal("https://github.com/atom/electron/issues");
            },
          },
        ],
      },
    ];

    return templateDefault;
  }
}

exports.MenuBuilder = MenuBuilder;
