import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";

const TRIGGER_API_URL = "https://api.trigger.dev/api/v1/tasks/ingest-pdf/trigger";

serve(async (req) => {
    try {
        const { pdfUrl } = await req.json();

        if (!pdfUrl) {
            return new Response(JSON.stringify({ error: "pdfUrl is required" }), { status: 400 });
        }

        const triggerKey = Deno.env.get("TRIGGER_SECRET_KEY");
        if (!triggerKey) {
            return new Response(JSON.stringify({ error: "TRIGGER_SECRET_KEY not set" }), { status: 500 });
        }

        // Trigger the background job
        const response = await fetch(TRIGGER_API_URL, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${triggerKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ payload: { pdfUrl } }),
        });

        const result = await response.json();

        return new Response(JSON.stringify({ message: "Ingestion started", jobId: result.id }), {
            headers: { "Content-Type": "application/json" },
        });
    } catch (error) {
        return new Response(JSON.stringify({ error: error.message }), { status: 500 });
    }
});
