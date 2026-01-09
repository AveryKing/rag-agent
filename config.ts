import dotenv from "dotenv";
import path from "path";

dotenv.config();

export const SUPABASE_URL = process.env.SUPABASE_URL || "";
export const SUPABASE_PRIVATE_KEY = process.env.SUPABASE_PRIVATE_KEY || "";

export const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
export const EMBEDDING_MODEL = "nomic-embed-text";
export const LLM_MODEL = "llama3";
export const DATABASE_URL = process.env.DATABASE_URL || "";

if (!SUPABASE_URL || !SUPABASE_PRIVATE_KEY || !DATABASE_URL) {
    console.warn("WARNING: Missing required environment variables (Supabase or Database URL).");
}
