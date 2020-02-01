import React from "react";
import folder from "./assets/folder.png";

function HotCorner({ handleClick }) {
  return (
    <div className="hotCorner" onClick={handleClick}>
      <img src={folder} />
    </div>
  );
}

export default HotCorner;
