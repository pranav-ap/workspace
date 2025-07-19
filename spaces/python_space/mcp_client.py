import ast
import asyncio
import pprint
import json

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

SERVER_URL = "http://localhost:8000/mcp"

pp = pprint.PrettyPrinter(indent=2, width=100)


def unwrap_tool_result(resp):
    """
    Safely unwraps the content from a FastMCP tool call result object.
    """
    if hasattr(resp, "content") and resp.content:
        content_object = resp.content[0]

        if hasattr(content_object, "json"):
            return content_object.model_dump_json()

        if hasattr(content_object, "text"):
            try:
                return ast.literal_eval(content_object.text())
            except (ValueError, SyntaxError):
                return content_object.text()

    return resp


async def main():
    print("\n🚀 Connecting to FastMCP server at:", SERVER_URL)
    transport = StreamableHttpTransport(url=SERVER_URL)
    client = Client(transport)

    async with client:
        print("\n🔗 Testing server connectivity...")
        await client.ping()
        print("✅ Server is reachable!\n")

        # Discover server capabilities
        print("🛠️  Available tools:")
        pp.pprint(await client.list_tools())
        print("\n📚 Available resources:")
        pp.pprint(await client.list_resources())
        print("\n💬 Available prompts:")
        pp.pprint(await client.list_prompts())

        # Fetch the topics resource
        # noinspection PyUnreachableCode
        if True:
            print("\n\n📖 Fetching resource: resource://ai/arxiv_topics")
            res = await client.read_resource("resource://ai/arxiv_topics")

            topics = ast.literal_eval(res[0].text)

            print("Today's AI topics:")
            for i, t in enumerate(topics, 1):
                print(f"  {i}. {t}")

        # Test the search tool
        # noinspection PyUnreachableCode
        if True:
            print("\n\n🔍 Testing tool: search_arxiv")
            search_results = await client.call_tool(
                "search_arxiv",
                {
                    "query": "Transformer interpretability",
                    "max_results": 3,
                },
            )

            search_results = unwrap_tool_result(search_results)
            search_results = json.loads(json.loads(search_results)['text'])

            for i, paper in enumerate(search_results, 1):
                print(f"  {i}. {paper['title']}\n     {paper['url']}")

            # Test the summarize tool on the first result
            if True and search_results:
                first_url = search_results[0]["url"]

                print("\n\n📝 Testing tool: summarize_paper")
                summary = await client.call_tool(
                    "summarize_paper",
                    {
                        "paper_url": first_url
                    }
                )

                summary = unwrap_tool_result(summary)
                summary = json.loads(summary)['text']
                print("\nSummary of first paper:\n", summary)

        # Test the prompt generator
        # noinspection PyUnreachableCode
        if True:
            print("\n\n🚀 Testing prompt: explore_topic_prompt")
            prompt_resp = await client.get_prompt(
                "explore_topic_prompt",
                {
                    "topic": "Transformer interpretability"
                }
            )

            print("\nGenerated prompt for an LLM:")
            for msg in prompt_resp.messages:
                print(f"{msg.role.upper()}: {msg.content.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
