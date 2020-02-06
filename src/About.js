import React, { useState, useEffect } from "react";
import FocusLock from "react-focus-lock";

import "./About.css";
var shell = window.require("electron").shell;
const settings = window.require("electron-settings");
import logo from "./assets/logo.png";

function About({ setAbout }) {
  const [keyInput, setKeyInput] = useState("");
  const [key, setKey] = useState(settings.get("licenseKey"));
  const [message, setMessage] = useState(false);

  const [registration, setRegistration] = useState(false);

  useEffect(() => {
    const getLicenseStatus = async () => {
      const response = await fetch(
        "https://api.gumroad.com/v2/licenses/verify",
        {
          body: `product_permalink=UVvUM&license_key=${key}`,
          headers: {
            "Content-Type": "application/x-www-form-urlencoded"
          },
          method: "POST"
        }
      );
      const license = await response.json();
      if (
        license.success === true &&
        license.purchase.refunded === false &&
        license.purchase.chargebacked === false &&
        license.purchase.disputed === false
      ) {
        setRegistration(true);
        settings.set("licenseKey", key);
      } else {
        setMessage(license.message);
      }
    };
    if (key && key.length > 0) {
      getLicenseStatus();
    }
  }, [key]);
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
            <FocusLock>
              <input
                type="text"
                placeholder="REGISTRATION KEY"
                onChange={e => setKeyInput(e.target.value)}
                value={keyInput}
              />
              <span className="message">{message}</span>
              <button
                onClick={() => {
                  setKey(keyInput);
                }}
              >
                Register
              </button>
            </FocusLock>
          </React.Fragment>
        )}
      </div>
    </div>
  );
}

export default About;
