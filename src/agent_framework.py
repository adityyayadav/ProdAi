import os
import re
from typing import Any, Dict, List, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from src.rag import get_random_by_category, get_ideas_db, _rag_categories
from src.youtube import search_youtube_tutorial, search_youtube_song
from src.logger import logger
from src.config import get_config


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output (e.g. Qwen)."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()



SPECIALISTS = {
    "composer": {
        "temperature": 0.7,
        "system_message": SystemMessage(content="You are an expert music composer and modern songwriter. Provide advice on composition, arrangement, and songwriting. Your specific influences include: Radiohead, Bloc Party, Sylvan Esso, Banks, As Tall as Lions, and Bon Iver.")
    },
    "mixing_engineer": {
        "temperature": 0.3,
        "system_message": SystemMessage(content="You are an expert mixing and mastering engineer. You studied under Nigel Godrich. Provide advice on mixing, EQ, compression, and other audio processing techniques.")
    },
    "sound_designer": {
        "temperature": 0.8,
        "system_message": SystemMessage(content="You are an expert sound designer and music producer. You studied under Nigel Godrich and Brian Eno. Provide advice on synthesizer programming, sample manipulation, and creating unique sounds.")
    },
    "lyricist": {
        "temperature": 0.7,
        "system_message": SystemMessage(content="You are an expert lyricist. Provide advice on writing compelling lyrics, developing themes and narratives, crafting hooks, and ensuring lyrics flow well with the music. Your specific influences include: Radiohead, Bloc Party, Sylvan Esso, Banks, As Tall as Lions, and Bon Iver. You enjoy obscure and opaque lyrics, but not everything needs to be complicated.")
    },
    "project_manager": {
        "temperature": 0.4,
        "system_message": SystemMessage(content="You are a music project manager. Coordinate the efforts and provide overall guidance on the music production process.")
    }
}

# Type definitions for our state
class AgentState(TypedDict):
    """State container for the agent workflow.
    
    Attributes:
        messages: Sequence of conversation messages
        previous_messages: Sequence of previous conversation messages
        current_agent: Currently selected specialist agent
        rag_results: Retrieved context from vector store
        needs_song: Flag indicating if a song example is needed
        suggested_song: Recommended song for reference
        youtube_url: YouTube URL for the suggested song
        is_random: Flag indicating if user requested a random idea
        is_followup: Flag indicating if this is a followup question
    """
    messages: Sequence[BaseMessage]
    previous_messages: Sequence[BaseMessage]
    current_agent: str
    rag_results: List[Any]
    needs_song: bool
    suggested_song: str | None
    youtube_url: str | None
    tutorial_url: str | None
    tutorial_title: str | None
    is_random: bool
    is_followup: bool


def get_best_groq_model(api_key: str) -> str:
    """Dynamically query Groq to pick an active model supported by your account."""
    explicit_model = os.getenv("GROQ_MODEL")
    if explicit_model:
        return explicit_model

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=5.0)
        models_data = client.models.list().data
        available = [m.id for m in models_data]
        logger.info(f"Available Groq models on your account: {available}")

        preferred = [
            # Top-tier chat models
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "llama-3.3-70b-versatile",
            "llama-3.3-70b-specdec",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b",
            # Qwen models (good chat capability)
            "qwen/qwen3.8-32b",
            "qwen/qwen3.8-27b",
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen-2.5-32b-instruct",
            # OpenAI-compatible OSS models
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            # Older but functional
            "llama-3.2-11b-vision-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-1b-preview",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
        for p in preferred:
            if p in available:
                logger.info(f"Auto-selected best Groq model: {p}")
                return p

        # Fallback: skip non-chat models (audio, safety, TTS, Arabic-only, etc.)
        skip_keywords = ["whisper", "guard", "embed", "tts", "orpheus", "allam", "safeguard"]
        for m in available:
            if not any(x in m.lower() for x in skip_keywords):
                logger.info(f"Auto-selected fallback Groq model: {m}")
                return m
    except Exception as e:
        logger.warning(f"Could not auto-detect Groq models from API: {e}")

    return "llama-3.1-8b-instant"


# Initialize our LLM (Groq LPU)
config = get_config()
groq_model = get_best_groq_model(config.groq_api_key)
try:
    logger.info(f"Initializing Groq LLM ({groq_model})")
    llm = ChatOpenAI(
        model=groq_model,
        api_key=config.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.7,
        max_tokens=350,
        timeout=15.0
    )
except Exception as e:
    logger.error(f"Error initializing Groq LLM: {e}")
    raise e

def is_random_request(state: Dict) -> Dict:
    """Check if the user is requesting a random idea."""
    last_message = state["messages"][-1].content.lower()
    is_random = "random idea" in last_message
    return {**state, "is_random": is_random}


async def get_random_idea_rag(state: Dict) -> Dict:
    """Handle requests for random ideas."""
    last_message = state["messages"][-1].content.lower()
    category = None
    
    categories_in_message = [category for category in _rag_categories if category.lower() in last_message]
    if categories_in_message:
        category = categories_in_message
    
    result = await get_random_by_category(category)
    if result:
        return {**state, "rag_results": [result]}
    return state


async def query_vectorstore(state: Dict) -> Dict:
    """Query the vectorstore and add results to state."""
    last_message = state["messages"][-1].content
    results = await get_ideas_db(last_message, k=3)
    
    # Return updated state with all existing fields
    return {**state, "rag_results": results}


def filter_relevant_ideas(state: Dict) -> Dict:
    """Pass semantically retrieved RAG results directly to the specialist."""
    # ChromaDB has already retrieved the top-k most relevant techniques for this query
    return state


def select_specialist(state: Dict) -> Dict:
    """Select the most appropriate specialist for the query using keyword-first routing to conserve rate limits."""
    last_msg = state["messages"][-1].content.lower() if state["messages"] else ""
    
    # 1. Fast, zero-token keyword classification
    if any(k in last_msg for k in ["mix", "eq", "equaliz", "compress", "sidechain", "master", "reverb", "delay", "vocal", "level", "db", "gain", "pan"]):
        matched_agent = "mixing_engineer"
    elif any(k in last_msg for k in ["synth", "sound", "patch", "sample", "texture", "bass", "filter", "lfo", "oscillator", "ambient", "envelope", "drone"]):
        matched_agent = "sound_designer"
    elif any(k in last_msg for k in ["lyric", "rhyme", "words", "verse", "chorus", "bridge", "hook", "poetry", "metaphor", "theme"]):
        matched_agent = "lyricist"
    elif any(k in last_msg for k in ["chord", "melody", "harmony", "progression", "tempo", "scale", "bpm", "songwriting", "structure", "radiohead", "key"]):
        matched_agent = "composer"
    else:
        # 2. Only invoke LLM if keyword matching is ambiguous (with strict max_tokens=15)
        specialist_selector_prompt = ChatPromptTemplate.from_messages([
            SystemMessage("Choose the single best role: composer, mixing_engineer, sound_designer, lyricist, or project_manager. Reply ONLY with the role key."), 
            MessagesPlaceholder(variable_name="messages"),
        ])
        matched_agent = "project_manager"
        try:
            mini_llm = llm.bind(max_tokens=15)
            response = mini_llm.invoke(
                specialist_selector_prompt.format_messages(messages=state["previous_messages"] + state["messages"])
            )
            cleaned = strip_think_tags(response.content)
            agent_key = cleaned.strip().lower().replace("*", "").replace(".", "").replace('"', '').replace("'", "")
            for spec in SPECIALISTS:
                if spec in agent_key:
                    matched_agent = spec
                    break
        except Exception as e:
            logger.warning(f"Error in select_specialist LLM call, using project_manager fallback: {e}")
            
    logger.info(f"Selected specialist: {matched_agent}")
    return {**state, "current_agent": matched_agent}


def specialist_response(state: Dict) -> Dict:
    """Generate a response from the selected specialist agent."""
    specialist_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="previous_messages"),
        ("user", "Query: {query}")
    ])

    try:
        specialist_llm = llm.bind(max_tokens=300)
        response = specialist_llm.invoke(
            specialist_prompt.format_messages(
                query=state["messages"][-1].content,
                agent_scratchpad=[SPECIALISTS[state["current_agent"]]["system_message"]],
                previous_messages=state["previous_messages"]
            )
        )
        content = strip_think_tags(response.content)
    except Exception as e:
        logger.error(f"Error calling specialist LLM: {e}")
        content = f"I'm your **{state['current_agent'].replace('_', ' ').title()}**. Here are the relevant production techniques found above from the knowledge base, or please try asking again in a few seconds!"

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=content)]
    }


def needs_song_suggestion(state: Dict) -> bool:
    """Determine if a song suggestion is needed based on the specialist's response."""
    # Always include reference song suggestions without wasting an extra LLM call
    return {**state, "needs_song": True}


async def get_song_suggestion(state: Dict) -> Dict:
    """Get video tutorial and reference song suggestion tailored directly to the user's request."""
    song = None
    youtube_url = None
    tutorial_url = None
    tutorial_title = None

    rag_results = state.get("rag_results", [])
    user_query = state["messages"][-1].content if state.get("messages") else ""
    specialist_text = state["messages"][-1].content if len(state.get("messages", [])) > 1 else ""

    # 1. Ask model to identify the most accurate song and tutorial topic matching the user's query
    song_prompt = ChatPromptTemplate.from_messages([
        SystemMessage('You are a music producer assistant. Identify 1 best reference song and 1 specific tutorial topic that directly answer the user request. Reply strictly in this format:\nSONG: Artist - Title | TUTORIAL: Topic'), 
        ("user", "User asked: {user_query}\nSpecialist advice: {specialist_response}")
    ])

    tutorial_topic = user_query
    try:
        mini_llm = llm.bind(max_tokens=60)
        res = mini_llm.invoke(
            song_prompt.format(user_query=user_query, specialist_response=specialist_text[:300])
        )
        content = strip_think_tags(res.content.strip())
        logger.info(f"Song/Tutorial extractor output: {content}")

        # Parse SONG
        if "SONG:" in content:
            song_part = content.split("SONG:")[1]
            if "|" in song_part:
                song = song_part.split("|")[0].strip()
            elif "TUTORIAL:" in song_part:
                song = song_part.split("TUTORIAL:")[0].strip()
            else:
                song = song_part.split("\n")[0].strip()

        # Parse TUTORIAL
        if "TUTORIAL:" in content:
            tutorial_topic = content.split("TUTORIAL:")[1].strip().split("\n")[0].strip()

    except Exception as e:
        logger.warning(f"Error extracting song/tutorial via LLM: {e}")

    # Fallback to RAG document if LLM produced nothing
    if not song and rag_results:
        for doc in rag_results:
            cand = getattr(doc, "Song", None)
            if cand and isinstance(cand, str) and len(cand.strip()) > 2 and cand.lower() != "none":
                song = cand.strip()
                break

    # 2. Search for the practical How-To Video Tutorial
    if tutorial_topic:
        try:
            tut = await search_youtube_tutorial(tutorial_topic)
            if tut:
                tutorial_url = tut.get("url")
                tutorial_title = tut.get("title")
        except Exception as e:
            logger.warning(f"Could not search YouTube tutorial: {e}")

    # 3. Search for the Reference Song
    if song:
        try:
            youtube_url = await search_youtube_song(song)
        except Exception as e:
            logger.warning(f"Could not search YouTube for song '{song}': {e}")
    
    return {
        **state, 
        "suggested_song": song, 
        "youtube_url": youtube_url,
        "tutorial_url": tutorial_url,
        "tutorial_title": tutorial_title
    }

def is_followup_question(state: Dict) -> Dict:
    """Check if this is a followup question based on message history length."""
    is_followup = len(state["previous_messages"]) > 0
    logger.info(f"Is followup question: {is_followup}")
    return {**state, "is_followup": is_followup}

# Graph construction
def create_agent_graph() -> StateGraph:
    """Create the workflow graph for the agent framework.
    
    The graph defines the following workflow:
    1. Check if user requested random idea
    2. Either get random idea or select specialist
    3. Query vector store for relevant context
    4. Filter results for relevance
    5. Generate specialist response
    6. Optionally suggest reference song
    
    Returns:
        Compiled workflow graph
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("is_followup", is_followup_question)
    workflow.add_node("is_random_request", is_random_request)
    workflow.add_node("get_random_idea_rag", get_random_idea_rag)
    workflow.add_node("select_specialist", select_specialist)
    workflow.add_node("query_rag", query_vectorstore)
    workflow.add_node("filter_rag", filter_relevant_ideas)
    workflow.add_node("specialist", specialist_response)
    workflow.add_node("needs_song", needs_song_suggestion)
    workflow.add_node("get_song_and_link", get_song_suggestion)
    
    # Connect the nodes
    workflow.add_edge(START, "is_followup")
    workflow.add_edge("get_random_idea_rag", END)
    workflow.add_edge("query_rag", "filter_rag")
    workflow.add_edge("filter_rag", "select_specialist")
    workflow.add_edge("select_specialist", "specialist")
    workflow.add_edge("specialist", "needs_song")
    workflow.add_edge("get_song_and_link", END)

    # Define the conditional edges
    workflow.add_conditional_edges(
        "is_followup",
        lambda state: state["is_followup"],
        {
            True: "select_specialist",  
            False: "is_random_request"  
        }
    )
    
    workflow.add_conditional_edges(
        "is_random_request",
        lambda state: state["is_random"],
        {
            True: "get_random_idea_rag",
            False: "query_rag"
        }
    )
    
    workflow.add_conditional_edges(
        "needs_song",
        lambda state: state["needs_song"],
        {
            True: "get_song_and_link",
            False: END
        }
    )
    
    return workflow.compile()

# Update the invoke_agent function to be async
async def invoke_agent(message: str, previous_messages: Optional[Sequence[BaseMessage]] = None) -> Dict:
    """Main entry point for the agent framework.
    
    Args:
        message: User input message
        previous_messages: Optional list of previous conversation messages
        
    Returns:
        Final state containing agent response and conversation history
    """
    graph = create_agent_graph()

    initial_state = AgentState(
        messages=[HumanMessage(content=message)],
        previous_messages=list(previous_messages) if previous_messages else [],
        current_agent="",
        rag_results=[],
        needs_song=False,
        suggested_song=None,
        youtube_url=None,
        tutorial_url=None,
        tutorial_title=None,
        is_random=False,
        is_followup=False
    )
    
    result = await graph.ainvoke(initial_state)

    # dont add previous messages if they are only requests for RAG
    if any(isinstance(msg, AIMessage) for msg in result["messages"]):
        result["previous_messages"] += result["messages"]
    return result

