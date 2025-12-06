<script setup>
import { ref, onMounted } from 'vue';
import { pipeline, env } from '@huggingface/transformers';
import Button from 'primevue/button';
import Textarea from 'primevue/textarea';
import InputText from 'primevue/inputtext';
import Card from 'primevue/card';
import ProgressBar from 'primevue/progressbar';
import Message from 'primevue/message';

// Configure transformers.js to use the hosted models (it will cache them in browser cache)
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = './public/models/';
console.log("Transformers Env:", env);

const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';

const loadingStatus = ref('Initializing...');
const isLoading = ref(true);
const progress = ref(0);
const extractor = ref(null);

const inputText = ref('');
const searchQuery = ref('');
const searchResults = ref([]);
const storedItems = ref([]);

onMounted(async () => {
    loadStoredItems();
    try {
        extractor.value = await pipeline('feature-extraction', MODEL_NAME, {
            quantized: true,
            local_files_only: true,
            subfolder: '',
            progress_callback: (data) => {
                if (data.status === 'progress') {
                    progress.value = data.progress;
                    loadingStatus.value = `Downloading model... ${Math.round(data.progress)}%`;
                } else if (data.status === 'initiate') {
                    loadingStatus.value = `Initiating download for ${data.file}...`;
                } else if (data.status === 'done') {
                    loadingStatus.value = `Model loaded from cache/downloaded.`;
                }
            }
        });
        isLoading.value = false;
        loadingStatus.value = 'Model ready.';
    } catch (error) {
        console.error("Error loading model:", error);
        loadingStatus.value = `Error loading model: ${error.message}`;
    }
});

const loadStoredItems = () => {
    const items = localStorage.getItem('semantic_search_items');
    if (items) {
        storedItems.value = JSON.parse(items);
    }
};

const saveMessage = async () => {
    if (!inputText.value.trim() || !extractor.value) return;

    const text = inputText.value.trim();
    
    try {
        const output = await extractor.value(text, { pooling: 'mean', normalize: true });
        const embedding = Array.from(output.data);

        const newItem = {
            id: Date.now(),
            text: text,
            embedding: embedding
        };

        storedItems.value.push(newItem);
        localStorage.setItem('semantic_search_items', JSON.stringify(storedItems.value));
        inputText.value = '';
    } catch (e) {
        console.error("Error generating embedding:", e);
    }
};

const cosineSimilarity = (a, b) => {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
};

const search = async () => {
    if (!searchQuery.value.trim() || !extractor.value) {
        searchResults.value = [];
        return;
    }

    const query = searchQuery.value.trim();
    
    try {
        const output = await extractor.value(query, { pooling: 'mean', normalize: true });
        const queryEmbedding = Array.from(output.data);

        const results = storedItems.value.map(item => {
            return {
                ...item,
                score: cosineSimilarity(queryEmbedding, item.embedding)
            };
        });

        // Sort by score descending
        results.sort((a, b) => b.score - a.score);
        searchResults.value = results;
    } catch (e) {
        console.error("Error searching:", e);
    }
};
</script>

<template>
    <div class="container">
        <div class="status-bar" v-if="isLoading || loadingStatus">
            <Message :severity="isLoading ? 'info' : 'success'" :closable="false">{{ loadingStatus }}</Message>
            <ProgressBar v-if="isLoading && progress > 0" :value="progress" class="mt-2"></ProgressBar>
        </div>

        <div class="grid">
            <div class="col-12 md:col-6">
                <Card>
                    <template #title>Add Data</template>
                    <template #content>
                        <div class="flex flex-column gap-2">
                            <label for="input-text">Enter text to save</label>
                            <Textarea id="input-text" v-model="inputText" rows="5" autoResize class="w-full" />
                            <Button label="Save" icon="pi pi-save" @click="saveMessage" :disabled="isLoading || !inputText" />
                        </div>
                    </template>
                </Card>
            </div>

            <div class="col-12 md:col-6">
                <Card>
                    <template #title>Semantic Search</template>
                    <template #content>
                        <div class="flex flex-column gap-2">
                            <label for="search-input">Search query</label>
                            <div class="p-inputgroup">
                                <InputText id="search-input" v-model="searchQuery" @keyup.enter="search" placeholder="Type and press enter..." />
                                <Button icon="pi pi-search" @click="search" :disabled="isLoading" />
                            </div>
                        </div>

                        <div class="results mt-4" v-if="searchResults.length > 0">
                            <h3>Results</h3>
                            <div v-for="result in searchResults" :key="result.id" class="result-item mb-2 p-2 border-round surface-ground">
                                <div class="font-bold">{{ result.text }}</div>
                                <div class="text-sm text-500">Similarity: {{ result.score.toFixed(4) }}</div>
                            </div>
                        </div>
                    </template>
                </Card>
            </div>
        </div>
        
        <div class="mt-4">
            <h3>Stored Items ({{ storedItems.length }})</h3>
            <div class="flex flex-wrap gap-2">
                 <span v-for="item in storedItems" :key="item.id" class="p-tag p-tag-info">{{ item.text.substring(0, 20) }}...</span>
            </div>
        </div>
    </div>
</template>

<style scoped>
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0rem;
}
.mt-2 { margin-top: 0.5rem; }
.mt-4 { margin-top: 2rem; }
.mb-2 { margin-bottom: 0.5rem; }
.p-2 { padding: 0.5rem; }
.w-full { width: 100%; }
.flex { display: flex; }
.flex-column { flex-direction: column; }
.gap-2 { gap: 0.5rem; }
.grid { display: grid; grid-template-columns: 1fr; gap: 2rem; }
@media (min-width: 768px) {
    .grid { grid-template-columns: 1fr 1fr; }
}
.surface-ground { background-color: var(--surface-ground); }
.border-round { border-radius: 6px; }
</style>
