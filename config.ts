import dotenv from "dotenv";
import path from "path";

dotenv.config();

export const SUPABASE_URL = process.env.SUPABASE_URL || "";
export const SUPABASE_PRIVATE_KEY = process.env.SUPABASE_PRIVATE_KEY || "";

export const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
export const EMBEDDING_MODEL = "nomic-embed-text";
export const LLM_MODEL = "llama3";

if (!SUPABASE_URL || !SUPABASE_PRIVATE_KEY) {
    console.warn("WARNING: SUPABASE_URL or SUPABASE_PRIVATE_KEY is not set.");
}
