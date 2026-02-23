import asyncio
import os
from temporalio.client import Client
from temporal_worker import LegislatorOrchestratorWorkflow

async def main():
    temporal_addr = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    client = await Client.connect(temporal_addr)

    legislator_name = "Rep_Suhas_Subramanyam"
    search_query = "Rep. Suhas Subramanyam"
    max_results = 5

    # 3. Execute the workflow
    # This 'starts' the workflow and returns a handle. 
    # 'result()' waits for the entire pipeline to finish.
    print(f"Starting pipeline for {legislator_name}...")
    
    handle = await client.start_workflow(
        LegislatorOrchestratorWorkflow.run,
        args=[legislator_name, search_query, max_results],
        id=f"workflow-{legislator_name}",
        task_queue="legislator-io",
    )

    print(f"Workflow started! ID: {handle.id}, Run ID: {handle.result_run_id}")
    
    # Optional: Wait for the result
    # result = await handle.result()
    # print("Workflow complete!")

if __name__ == "__main__":
    asyncio.run(main())