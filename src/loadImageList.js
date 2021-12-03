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

export default async function loadImageList() {
  const items = await knex().select("*").from("things")
  return { items: items.map((item) => ({ fileName: item.uri, modified: 0 })) };
}
