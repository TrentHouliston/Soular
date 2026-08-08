/** @type {import("prettier").Config} */
module.exports = {
  arrowParens: "always",
  bracketSpacing: true,
  endOfLine: "lf",
  jsonRecursiveSort: true,
  plugins: ["prettier-plugin-sort-json"],
  printWidth: 120,
  proseWrap: "preserve",
  quoteProps: "as-needed",
  semi: true,
  singleQuote: false,
  tabWidth: 2,
  trailingComma: "es5",
  useTabs: false,
  overrides: [
    {
      // Home Assistant manifest.json - domain and name first, then alphabetical
      // See: https://github.com/home-assistant/core/blob/dev/script/util.py
      files: "custom_components/soular/manifest.json",
      options: {
        jsonSortOrder: JSON.stringify({
          domain: null,
          name: null,
          "/.*/": "lexical",
        }),
      },
    },
    {
      // HACS json files
      files: "hacs.json",
      options: {
        jsonSortOrder: JSON.stringify({
          name: null,
          hacs: null,
          homeassistant: null,
          "/.*/": "lexical",
        }),
      },
    },
    {
      // Home Assistant translation files - semantic ordering for config flows
      files: "custom_components/soular/translations/*.json",
      options: {
        objectWrap: "collapse",
        jsonSortOrder: JSON.stringify({
          // Flow-level keys (config, options, config_subentries.*)
          config: null,
          options: null,
          config_subentries: null,
          flow_title: null,
          entry_type: null,
          initiate_flow: null,
          step: null,
          error: null,
          abort: null,
          progress: null,
          create_entry: null,

          // Step/issue content keys
          title: null,
          description: null,
          data: null,
          data_description: null,

          // Initiate flow keys
          user: null,
          reconfigure: null,

          // Issues-specific key
          fix_flow: null,

          // Everything else sorted lexically (alphabetically)
          "/.*/": "lexical",
        }),
      },
    },
  ],
};
