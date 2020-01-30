import React from "react";
import gear from "./assets/gear.png";

function HotCorner({ handleClick }) {
  return (
    <div className="settingsButton" onClick={handleClick}>
      <img src={gear} />
    </div>
  );
}

export default HotCorner;
