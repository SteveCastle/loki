import React, { useState, useEffect, useRef } from "react";
import ReactDOM from "react-dom";
import "babel-polyfill";
import "./style.css";
import HotKeyController from "./HotKeyController";

import Detail from "./Detail";
import List from "./List";

// NODE IMPORTS
const settings = window.require("electron-settings");
const atob = window.require("atob");

import { SORT, FILTER, SIZE, VIEW } from "./constants";
import loadImageList from "./loadImageList";

function App() {
  const [view, setView] = useState(VIEW.DETAIL);
  const [path, setPath] = useState(atob(window.location.search.substr(1)));
  const [items, setItems] = useState([]);
  const [cursor, setCursor] = useState(0);
  const [sort, setSort] = useState(SORT[settings.get("settings.defaultSort")]);
  const [filter, setFilter] = useState(FILTER.ALL);
  const [size, setSize] = useState(SIZE.COVER);
  const [recursive, setRecursive] = useState(false);

  // Initialize State from settings.
  useEffect(() => {
    if (settings.has("settings.scaleMode")) {
      setSize(SIZE[settings.get("settings.scaleMode")]);
    }
  }, []);
  // Reload data from image provider if directory, filter, or recursive setting changes
  useEffect(() => {
    async function fetchData() {
      const data = await loadImageList({
        path,
        filter,
        sortOrder: sort,
        recursive
      });
      setItems(data.items);
      setCursor(data.cursor);
      console.log(data);
    }
    fetchData();
  }, [sort, filter, recursive]);

  function handleClick(e) {
    // If click is within 70 pixels of the bottom left hand corner display the list view.
    e.preventDefault();
    if (window.innerWidth - e.pageX < 70 && window.innerHeight - e.pageY < 70) {
      setView(VIEW.LIST);
      return 0;
    }

    // If click is within 70 pixels of the bottom right hand corner display the list view.
    if (window.innerWidth - e.pageX > 70 && window.innerHeight - e.pageY < 70) {
      setView(VIEW.LIST);
      return 0;
    }
    //If click is on the left decrease the cursor, if it is on the left increase it.
    e.pageX > window.innerWidth / 2
      ? setCursor(cursor === items.length - 1 ? cursor : cursor + 1)
      : setCursor(cursor === 0 ? cursor : cursor - 1);
  }

  function handleKeyPress(e) {
    switch (e.key) {
      case "s":
        e.preventDefault();
        setSort(sort === SORT.CREATE_DATE ? SORT.ALPHA : SORT.CREATE_DATE);
        break;
      case "c":
        e.preventDefault();
        setSize(size === SIZE.ACTUAL ? SIZE.COVER : SIZE.ACTUAL);
        break;
      case "r":
        e.preventDefault();
        setRecursive(!recursive);
        break;
      case "g":
        e.preventDefault();
        setFilter(FILTER.GIF);
        break;
      case "a":
        e.preventDefault();
        setFilter(FILTER.ALL);
        break;
      case "v":
        e.preventDefault();
        setFilter(FILTER.VIDEO);
        break;
      case "j":
        e.preventDefault();
        setFilter(FILTER.STATIC);
        break;
      default:
        e.preventDefault();
        console.log(`pressed ${e.key}`);
    }
  }

  if (items.length === 0) {
    return (
      <div
        tabIndex="0"
        onKeyPress={handleKeyPress}
        className="noItemsContainer"
      >
        <h1>NO ITEMS FOUND</h1>
      </div>
    );
  }
  switch (view) {
    case VIEW.DETAIL:
      return (
        <React.Fragment>
          <HotKeyController handleKeyPress={handleKeyPress} />
          <Detail
            fileName={items[cursor].fileName}
            size={size}
            handleClick={handleClick}
          />
        </React.Fragment>
      );
    case VIEW.LIST:
      return (
        <React.Fragment>
          <HotKeyController handleKeyPress={handleKeyPress} />
          <List
            fileList={items}
            size={size}
            cursor={cursor}
            handleClick={i => {
              setCursor(i);
              setView(VIEW.DETAIL);
            }}
          />
        </React.Fragment>
      );
    default:
      return (
        <List
          fileList={items}
          size={size}
          cursor={cursor}
          handleClick={i => {
            setCursor(i);
            setView(VIEW.DETAIL);
          }}
        />
      );
  }
}
ReactDOM.render(<App />, document.getElementById("root"));
