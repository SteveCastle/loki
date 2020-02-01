import React from "react";
import folder from "./assets/folder.png";

function HotCorner({ handleClick }) {
  return (
    <div
      className="hotCorner"
      onClick={handleClick}
      disabled="disabled"
      tabIndex="-1"
    >
      <img src={folder} />
    </div>
  );
}

export default HotCorner;
