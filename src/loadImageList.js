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

export default async function loadImageList(filters) {

  const items = await knex("things_concepts")
    .join("things", "things_concepts.things_id", "things.id")
    .join("labels", "labels.concepts_id", "things_concepts.concepts_id")
    .select("things.*")
    .whereRaw(
      `things_concepts.concepts_id = labels.concepts_id AND (labels.literal IN (${filters.map(_ => '?').join(',')})) AND things.id = things_concepts.things_id`
    , filters)
    .groupBy("things.id")
    .havingRaw("COUNT(things.id)=?", filters.length).debug();
  return { items: items.map((item) => ({ fileName: item.uri, modified: 0 })) };
}
