import React, { useState } from "react";
import "./About.css";
var shell = window.require("electron").shell;
const settings = window.require("electron-settings");
import logo from "./assets/logo.png";

function About({ setAbout }) {
  const [key, setKey] = useState("");
  const [message, setMessage] = useState(false);

  const [registration, setRegistration] = useState(
    settings.get("settings.registration")
  );
  return (
    <div className="aboutContainer">
      <div className="aboutMenu">
        <div>
          <img src={logo} />
          <span
            className="closeButton"
            onClick={() => {
              if (registration) {
                setAbout(false);
              } else {
                setMessage(
                  "Please register to support development of LowKey Image Viewer and hide this message."
                );
              }
            }}
          >
            ✖
          </span>
          <h1>LowKey Image Viewer</h1>
        </div>
        <span className="version">
          Version: 1.0.5
          <span className="updateCheck">(Check for Updates)</span>
        </span>
        <div className="registrationStatusContainer">
          <span className="registrationStatus">
            {`Registration Status: ${
              registration ? "Registered" : "Unregistered"
            }`}
          </span>
          {!registration && (
            <span
              className="purchaseLink"
              onClick={() => {
                shell.openExternal("https://gumroad.com");
              }}
            >
              (Buy now on GumRoad)
            </span>
          )}
        </div>
        {!registration && (
          <React.Fragment>
            <input
              type="text"
              placeholder="REGISTRATION KEY"
              onChange={e => setKey(e.target.value)}
              value={key}
            />
            <span className="message">{message}</span>
            <button
              onClick={() => {
                setRegistration(true);
                setAbout(false);
              }}
            >
              Register
            </button>
          </React.Fragment>
        )}
      </div>
    </div>
  );
}

export default About;
