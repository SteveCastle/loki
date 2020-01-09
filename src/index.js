import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom";
import "babel-polyfill";

// NODE IMPORTS
const url = window.require("url");
const settings = window.require("electron-settings");
const atob = window.require("atob");

import { SORT, FILTER, SIZE } from "./constants";
import loadImageList from "./loadImageList";

function App() {
  const [path, setPath] = useState(atob(window.location.pathname.substr(1)));
  const [items, setItems] = useState([]);
  const [sort, setSort] = useState(SORT.ALPHA);
  const [filter, seteFilter] = useState(FILTER.ALL);
  const [size, setSize] = useState(SIZE.COVER);
  const [recursive, setRecursive] = useState(false);

  useEffect(() => {
    async function fetchData() {
      const data = await loadImageList({
        path,
        filter,
        sort,
        recursive
      });
      setItems(data);
      console.log(data);
    }
    fetchData();
  }, [sort, filter, recursive]);
  return (
    <div>
      <h1>Settings {settings.get("name.first")}</h1>
      <h2>Location: {atob(window.location.pathname.substr(1))}</h2>
      {items.map(item => (
        <img
          key={item.patfileNameh}
          src={url.format({
            protocol: "file",
            pathname: item.fileName
          })}
        />
      ))}
    </div>
  );
}
ReactDOM.render(<App />, document.getElementById("root"));
