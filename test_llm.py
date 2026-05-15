import asyncio
from src.llm_client import query_large

async def main():
    prompt = """You are a script annotator. Your task is to identify who is speaking in the dialogue segments of the provided text.

CRITICAL INSTRUCTION: Only select speakers who are actively present in the scene. Use the provided character biographies to match the speaker's description, location, and personality with the current scene's context. Do not select characters who are not physically present.

VALID SPEAKERS AND BIOS:
nova_pilot: Nova - A skilled starship pilot
zane_mechanic: Zane - A cynical cyborg mechanic

RAW TEXT:
“I don’t have enough fuel,” replied Nova cheerfully. “My tank is almost empty, you know, and that is why I am going to Alpha Centauri to ask for some supplies.”
“Oh, I see,” said Zane. “But, after all, fuel isn't the most important thing in the galaxy.”

SLICED FRAGMENTS:
0 [D]: “I don’t have enough fuel,”
1 [T]: replied Nova cheerfully. 
2 [D]: “My tank is almost empty, you know, and that is why I am going to Alpha Centauri to ask for some supplies.”
3 [N]: 

4 [D]: “Oh, I see,”
5 [T]:  said Zane. 
6 [D]: “But, after all, fuel isn't the most important thing in the galaxy.”

TASK:
For every segment labeled [D], identify the correct speaker_id from the VALID SPEAKERS list based on their bio and the text.
Use the RAW TEXT to understand the context and dialogue tags.

OUTPUT FORMAT:
Only output the mapping as 'index: speaker_id'. One per line. No preamble.

Example:
1: nova_pilot
3: zane_mechanic
"""
    res = await query_large(prompt)
    print("RAW RESPONSE:")
    print(res)

asyncio.run(main())
