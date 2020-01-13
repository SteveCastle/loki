import React, { Component } from "react";
import { FixedSizeGrid as Grid } from "react-window";
const url = require("url");
import ListItem from "./ListItem";
const Cell = ({ data, columnIndex, rowIndex, style }) =>
  data.fileList[rowIndex * data.columns + columnIndex] ? (
    <div
      style={{
        ...style,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        overflow: "hidden"
      }}
    >
      <ListItem
        className="listImage"
        handleClick={() =>
          data.handleClick(rowIndex * data.columns + columnIndex)
        }
        fileName={
          data.fileList[rowIndex * data.columns + columnIndex] &&
          data.fileList[rowIndex * data.columns + columnIndex].fileName
        }
      />
    </div>
  ) : (
    <div></div>
  );

export default class List extends Component {
  gridRef = null;
  rows = this.props.tall
    ? Math.ceil(this.props.fileList.length / 3)
    : Math.ceil(Math.sqrt(this.props.fileList.length));
  columns = this.props.tall
    ? 3
    : Math.ceil(Math.sqrt(this.props.fileList.length));
  componentDidMount() {
    this.gridRef.scrollToItem({
      columnIndex: this.props.cursor % this.columns,

      rowIndex: Math.floor(this.props.cursor / this.columns)
    });
  }
  render() {
    const { fileList, handleClick } = this.props;
    return (
      <div className="container" data-tid="container">
        <Grid
          ref={r => {
            this.gridRef = r;
          }}
          columnCount={this.columns}
          columnWidth={window.innerWidth / 3}
          height={window.innerHeight}
          rowCount={this.rows}
          rowHeight={600}
          width={window.innerWidth}
          itemData={{ fileList, handleClick, columns: this.columns }}
        >
          {Cell}
        </Grid>
      </div>
    );
  }
}
