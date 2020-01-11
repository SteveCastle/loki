import React, { useRef, useEffect } from "react";
const url = window.require("url");
const path = window.require("path");

import { SIZE, EXTENSIONS } from "./constants";

function Detail({ fileName, size, handleClick }) {
  const ref = useRef(null);

  return (
    <div className="container" onClick={handleClick} tabIndex="0">
      {EXTENSIONS.img.includes(path.extname(fileName)) && (
        <img
          key={fileName}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          className={`${size === SIZE.COVER ? "cover" : null}`}
        />
      )}
      {EXTENSIONS.video.includes(path.extname(fileName)) && (
        <video
          className={`${size === SIZE.COVER ? "cover" : null}`}
          src={url.format({
            protocol: "file",
            pathname: fileName
          })}
          loop
          autoPlay
          controls
        />
      )}
    </div>
  );
}

export default Detail;
