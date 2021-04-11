import React from "react";
var shell = window.require("electron").shell;
import "./About.css";
var shell = window.require("electron").shell;
const settings = window.require("electron-settings");
import logo from "./assets/logo.png";

function About({ setAbout }) {
  return (
    <div className="aboutContainer">
      <div className="aboutMenu">
        <div>
          <img src={logo} />
          <span
            className="closeButton"
            onClick={() => {
              setAbout(false);
            }}
          >
            ✖
          </span>
          <h1>LowKey Image Viewer</h1>
        </div>
        <span className="version">Version: 1.1.0</span>
        <span>
          <a
            href=""
            className="homePageLink"
            onClick={() => {
              shell.openExternal("https://lowkeyviewer.com");
            }}
          >
            Get Updates and Help
          </a>
        </span>
        <span className="donationReqeust">
          Thank you for using Lowkey Image Viewer. If you can please support
          development of this project by becoming a Patron on Patreon.
        </span>
        <button
          onClick={() => {
            shell.openExternal("https://patreon.com/lowkeyviewer");
          }}
        >
          Support on Patreon
        </button>
      </div>
    </div>
  );
}

export default About;
