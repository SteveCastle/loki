import React, { useEffect, useState } from "react";
import FocusLock from "react-focus-lock";
import "./TagFilters.css";

const knex = window.require("knex")({
  client: "pg",
  connection: {
    host: "127.0.0.1",
    port: 5432,
    user: "sparrow",
    password: "sparrow",
    database: "sparrow",
  },
});

function TagFilters({ value, handleChange, hide, path }) {
  const [currentTags, setCurrentTags] = useState([]);
  const [tags, setTags] = useState([]);
  useEffect(() => {
    async function loadTags() {
      const tags = await knex("labels").select("*");
      setTags(tags);
    }
    loadTags();
  }, []);
  useEffect(() => {
    async function loadCurrentTags() {
      const tags = await knex("things_concepts")
        .select("labels.*")
        .join("labels", "labels.concepts_id", "things_concepts.concepts_id")
        .join("things", "things.id", "things_concepts.things_id")
        .where({ uri: path }).debug();
        setCurrentTags(tags);
    }
    if(path){loadCurrentTags();}
  }, [path]);

  return (
    <div className="TagFilters">
      <div className="TagSearch">
        <FocusLock>
          <input value={value} onChange={(e) => handleChange(e.target.value)} />
        </FocusLock>
        <button onClick={hide}>Hide</button>
      </div>
      <div className="tagList">
      <ul className="suggestedTags">
        {tags
          .filter((tag) => tag.literal.startsWith(value.split(' ').slice(-1)[0]))
          .slice(0, 20)
          .map((tag) => (
            <li onClick={() => handleChange(` ${value.split(' ').slice(0, -1).join(" ")} ${tag.literal}`)}>{tag.literal}</li>
          ))}
      </ul>
      <ul className="currentTags">
        {currentTags.sort((a,b) => a.literal > b.literal ? 1 : - 1).map((tag) => (
          <li onClick={() => handleChange(tag.literal)}>{tag.literal} <span onClick={(e) => {
            e.stopPropagation()
            handleChange(`${value} ${tag.literal}`)}}>+</span></li>
        ))}
      </ul></div>
    </div>
  );
}

export default TagFilters;
