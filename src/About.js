import React from "react";
import "./About.css";
var shell = window.require("electron").shell;
import logo from "./assets/logo.png";

function About({ setAbout }) {
  return (
    <div className="aboutContainer">
      <div className="aboutMenu">
        <div>
          <img src={logo} />
          <span className="closeButton" onClick={() => setAbout(false)}>
            ✖
          </span>
          <h1>LowKey Image Viewer</h1>
        </div>
        <span className="version">
          Version: 0.0.1{" "}
          <span className="updateCheck">(Check for Updates)</span>
        </span>
        <div className="registrationStatusContainer">
          <span className="registrationStatus">
            Registration Status: Unregistered
          </span>
          <span
            className="purchaseLink"
            onClick={() => {
              shell.openExternal("https://gumroad.com");
            }}
          >
            (Buy now on GumRoad)
          </span>
        </div>
        <input type="text" placeholder="REGISTRATION KEY" />
        <button>Register</button>
      </div>
    </div>
  );
}

export default About;
