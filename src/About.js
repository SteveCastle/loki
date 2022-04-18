import React from "react";
var shell = window.require("electron").shell;
import "./About.css";
var shell = window.require("electron").shell;
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
        <span className="version">Version: 1.1.4-patron</span>
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
          Thank you for becoming a patron of Lowkey Image Viewer. 🎉
        </span>
      </div>
    </div>
  );
}

export default About;
