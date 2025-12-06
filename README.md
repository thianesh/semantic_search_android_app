# Semantic Search App

A Vue 3 application that performs semantic search using local embeddings via [@huggingface/transformers](https://huggingface.co/docs/transformers.js).

## Features

- **Local Embeddings**: Generates text embeddings locally in the browser using the `Xenova/all-MiniLM-L6-v2` model.
- **Semantic Search**: Performs vector similarity search using cosine similarity to find relevant messages.
- **Privacy Focused**: All data and processing happen client-side; no data is sent to external servers.
- **Persistence**: Saves messages and embeddings to the browser's Local Storage.
- **Modern UI**: Built with [PrimeVue](https://primevue.org/) components for a polished look and feel.

## Recommended IDE Setup

[VS Code](https://code.visualstudio.com/) + [Vue (Official)](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (and disable Vetur).

## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Compile and Minify for Production

```sh
npm run build
```
