import React from "react";

function Settings() {
  return (
    <div className="settingsContainer">
      <h2 className="settingsSectionHeader">Settings</h2>
      <h3>General</h3>
      <div className="widget">
        <label>Control Mode</label>
        <input />
      </div>
      <div className="widget">
        <label>Open Fullscreen</label>
        <input />
      </div>
      <div className="widget">
        <label>Keep Window on Top</label>
        <input />
      </div>
      <div className="widget">
        <label>Initial Sort Order</label>
        <input />
      </div>
      <div className="widget">
        <label>Item Filter</label>
        <input />
      </div>
      <h3 className="settingsSectionHeader">Image View</h3>
      <div className="widget">
        <label>Scaling Mode</label>
        <input />
      </div>

      <h3 className="settingsSectionHeader">List View</h3>
      <div className="widget">
        <label>Number of Columns</label>
        <input />
      </div>
      <div className="widget">
        <label>Item Size</label>
        <input />
      </div>
      <div className="widget">
        <label>Scaling Mode</label>
        <input />
      </div>
      <div className="widget">
        <label>Include Videos</label>
        <input />
      </div>
    </div>
  );
}

export default Settings;
