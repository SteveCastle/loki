import React from "react";

function HotCorner({ handleClick }) {
  return (
    <div className="settingsButton" onClick={handleClick}>
      ⚙️
    </div>
  );
}

export default HotCorner;
