import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'tests', timeout:60000, workers:1,
  use:{baseURL:'http://127.0.0.1:8014',channel:'chrome',headless:true, screenshot:'only-on-failure'},
  reporter:[['list']], outputDir:'../.runtime/browser-tests'});
