import React from "react";
const url = window.require("url");
import { SIZE } from "./constants";

function Detail({ fileName, size, handleClick, handleKeyPress }) {
  return (
    <div
      className="container"
      onClick={handleClick}
      tabIndex="0"
      onKeyPress={handleKeyPress}
    >
      <img
        key={fileName}
        src={url.format({
          protocol: "file",
          pathname: fileName
        })}
        className={`${size === SIZE.COVER ? "cover" : null}`}
      />
    </div>
  );
}

export default Detail;
