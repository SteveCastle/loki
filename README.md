# Loki: An Electron Image and Video Viewer

https://github.com/SteveCastle/loki/assets/1828509/f61d4aad-01b1-4bde-802a-876fd71f55ce


A minimalist web native image viewer project built with Electron and powering Lowkey Media Viewer.

Visit https://lowkeyviewer.com to download a prebuilt binary for Mac or Windows if you do not want to build it from source your self.

## Building from Source

> ### Requirements
>
> - Node 18
> - Yarn

You will also need to download exiftool, ffmpeg, ffplay, and ffprobe, and place them in `src/main/resources/bin`.


## Running in Development Mode

```
cd loki
yarn
yarn dev
```

## Building a Production Binary

To build a binary for your current operating system simply clone this repository and from the project root run yarn
to install dependencies, and yarn package to build the binary.

```
cd loki
yarn
yarn package
```

## Automated Releases

This project uses GitHub Actions for automated continuous integration and releases. When changes are pushed to the `main` branch:

1. All tests are run (Electron app and Go server)
2. Binaries are built for all platforms (macOS, Windows, Linux)
3. A new release is created with the version from `package.json`
4. All binaries are automatically uploaded to the release

To create a new release, simply update the version in `package.json` and push to `main`. See [.github/WORKFLOW.md](.github/WORKFLOW.md) for detailed documentation.

## Contributing

If you would like to contribute to loki please feel free to fork and make a pull request back to the master branch. If you fork this project
and make your own personal changes I'd also love to see your work so feel free to send me a message.
