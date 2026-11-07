const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const ts = require("typescript");
const source = fs.readFileSync(require("node:path").join(__dirname, "../src/hosted.ts"), "utf8");
function client(storage) {
  const context = { exports: {}, __HOSTED_DEMO__: true, sessionStorage: storage,
    fetch: async () => ({ ok: true, json: async () => ({ flows: [{ id: "f1" }], metrics: [] }) }) };
  vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, context);
  return context.exports.demoApi;
}
test("blocked storage gives an actionable error", async () => {
  const api = client({ getItem() { throw Error("SecurityError"); } });
  await assert.rejects(api("session", "POST"), /Allow site storage/);
});
test("duplicate persisted reviews are rejected", async () => {
  const api = client({ getItem: () => JSON.stringify([{flow_id:"f1",label:null},{flow_id:"f1",label:null}]) });
  await assert.rejects(api("reviews", "GET"), /invalid/);
});
test("failed writes preserve prior state and can be retried", async () => {
  let saved = null;
  let fail = true;
  const api = client({ getItem: () => saved, setItem(key,value) { if(fail) throw Error("QuotaExceededError"); saved=value; } });
  await assert.rejects(api("reviews/f1", "POST"), /could not be saved/);
  assert.equal((await api("reviews", "GET")).used, 0);
  fail=false;
  await api("reviews/f1", "POST");
  await api("reviews/f1", "POST");
  assert.equal((await api("reviews", "GET")).used, 1);
  await api("reviews/f1", "PUT", {label:"benign"});
  assert.equal((await api("reviews", "GET")).reviews[0].label, "benign");
});
