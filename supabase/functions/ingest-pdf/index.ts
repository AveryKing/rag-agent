import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const PROJECT_REF = "proj_gsodbjyoytbunyagnezk";
const TRIGGER_API_URL = `https://api.trigger.dev/api/v1/projects/${PROJECT_REF}/tasks/ingest-pdf/trigger`;

serve(async (req) => {
    // Handle CORS preflight
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders });
    }

    try {
        const { pdfUrl } = await req.json();

        if (!pdfUrl) {
            return new Response(JSON.stringify({ error: "pdfUrl is required" }), {
                status: 400,
                headers: { ...corsHeaders, "Content-Type": "application/json" }
            });
        }

        const triggerKey = Deno.env.get("TRIGGER_SECRET_KEY");
        if (!triggerKey) {
            return new Response(JSON.stringify({ error: "TRIGGER_SECRET_KEY not set in Supabase project secrets" }), {
                status: 500,
                headers: { ...corsHeaders, "Content-Type": "application/json" }
            });
        }

        // Trigger the background job via REST API
        console.log(`Triggering Trigger.dev task for: ${pdfUrl}`);
        const response = await fetch(TRIGGER_API_URL, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${triggerKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ payload: { pdfUrl } }),
        });

        if (!response.ok) {
            const errorData = await response.text();
            console.error(`Trigger.dev error: ${errorData}`);
            throw new Error(`Failed to trigger job: ${errorData}`);
        }

        const result = await response.json();

        return new Response(JSON.stringify({ message: "Ingestion started", jobId: result.id }), {
            headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
    } catch (error) {
        console.error(`Edge Function Error: ${error.message}`);
        return new Response(JSON.stringify({ error: error.message }), {
            status: 500,
            headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
    }
});
