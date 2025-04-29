import os
from typing import Dict, TypedDict, Annotated, List, Optional
from langgraph.graph import Graph, END, START
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from datetime import datetime, timedelta
import re

from dotenv import load_dotenv

from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup


load_dotenv()
chat_model = ChatGroq(groq_api_key = os.getenv('GROQ_API_KEY'), model_name = "Llama3-8b-8192")

## Data Structures
class GraphState(TypedDict):
    news_query: Annotated[str, "Input query to extract news search parameters from."]
    num_searches_remaining: Annotated[int, "Number of articles to search for."]
    newsapi_params: Annotated[dict, "Structured argument for the News API."]
    past_searches: Annotated[List[dict], "List of search params already used."]
    articles_metadata: Annotated[list[dict], "Article metadata response from the News API"]
    scraped_urls: Annotated[List[str], "List of urls already scraped."]
    num_articles_tldr: Annotated[int, "Number of articles to create summary for."]
    potential_articles: Annotated[List[dict[str, str, str]], "Article with full text to consider summarizing."]
    tldr_articles: Annotated[List[dict[str, str, str]], "Selected article summaries."]
    formatted_results: Annotated[str, "Formatted results to display."]

class NewsApiParams(BaseModel):
    q: str = Field(description="1-3 concise keyword search terms that are not too specific")
    #sources: Optional[str] = Field(None, description="Optional: comma-separated list of sources from available sources")
    from_param: str = Field(description="date in format 'YYYY-MM-DD' Two days ago minimum.")
    to: str = Field(description="date in format 'YYYY-MM-DD' today's date unless specified")
    language: str = Field(description="language of articles 'en' unless specified one of ['ar', 'de', 'en', 'es', 'fr', 'he', 'it', 'nl', 'no', 'pt', 'ru', 'se', 'ud', 'zh']")
    sort_by: str = Field(description="sort by 'relevancy', 'popularity', or 'publishedAt'")
    searchIn: Optional[List[str]] = Field(None, description="Optional: list of search fields- title, description, content")

## Graph Node functions
def generate_newsapi_params(state: GraphState) -> GraphState:
    """Based on the query, generate News API params."""
    # initialize parser to define the structure of the response
    #parser = JsonOutputParser(pydantic_object=NewsApiParams)
    parser = OutputFixingParser.from_llm(parser=JsonOutputParser(pydantic_object=NewsApiParams), llm=chat_model)

    # retrieve today's date
    today_date = datetime.now().strftime("%Y-%m-%d")

    # retrieve list of past search params
    past_searches = state["past_searches"]

    # retrieve number of searches remaining
    num_searches_remaining = state["num_searches_remaining"]

    # retrieve the user's query
    news_query = state["news_query"]

    print(state["newsapi_params"])

    # retrieve language
    lang = state["newsapi_params"]["language"]

    # retrieve sorting method
    sortby = state["newsapi_params"]["sort_by"]

    # retrieve search fields
    searchIn = state["newsapi_params"]["searchIn"]

    template = """
    Today is {today_date}.

    Create a param dict for the News API based on the user query. Ensure that field names are **exactly** as specified. Only return the param dict and no other messages or comments.
    q: {query}, language: {lang}, sort_by: {sortby}, searchIn: {searchIn}

    These searches have already been made. If required, loosen the search terms to get more results.
    {past_searches}
    
    Following these formatting instructions:
    {format_instructions}

    Including this one, you have {num_searches_remaining} searches remaining.
    If this is your last search, use all news sources and a 30 days search range.
    """

    # create a prompt template to merge the query, today's date, and the format instructions
    prompt_template = PromptTemplate(
        template=template,
        variables={"today": today_date, "query": news_query, "past_searches": past_searches, "num_searches_remaining": num_searches_remaining, "lang": lang, "sortby": sortby, "searchIn": searchIn},
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # create prompt chain template
    chain = prompt_template | chat_model | parser

    # invoke the chain with the news api query
    result = chain.invoke({"query": news_query, "today_date": today_date, "past_searches": past_searches, "num_searches_remaining": num_searches_remaining, "lang": lang, "sortby": sortby, "searchIn": searchIn})

    # update the state
    state["newsapi_params"] = result

    return state

def retrieve_articles_metadata(state: GraphState) -> GraphState:
    """Using the NewsAPI params, perform api call."""
    # parameters generated for the News API
    newsapi_params = state["newsapi_params"]

    # decrement the number of searches remaining
    state['num_searches_remaining'] -= 1

    try:
        # create a NewsApiClient object
        newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY'))
        
        # retreive the metadata of the new articles
        articles = newsapi.get_everything(**newsapi_params)

        # append this search term to the past searches to avoid duplicates
        state['past_searches'].append(newsapi_params)

        # load urls that have already been returned and scraped
        scraped_urls = state["scraped_urls"]

        # filter out articles that have already been scraped
        new_articles = []
        for article in articles['articles']:
            if article['url'] not in scraped_urls and len(state['potential_articles']) + len(new_articles) < 10:
                new_articles.append(article)

        # reassign new articles to the state
        state["articles_metadata"] = new_articles

    # handle exceptions
    except Exception as e:
        print(f"Error: {e}")

    return state

def retrieve_articles_text(state: GraphState) -> GraphState:
    """Web scrapes to retrieve article text."""
    # load retrieved article metadata
    articles_metadata = state["articles_metadata"]
    # Add headers to simulate a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
    }

    # create list to store valid article dicts
    potential_articles = []

    # iterate over the urls
    for article in articles_metadata:
        # extract the url
        url = article['url']

        # use beautiful soup to extract the article content
        response = requests.get(url, headers=headers)
        
        # check if the request was successful
        if response.status_code == 200:
            # parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # find the article content
            text = soup.get_text(strip=True)

            # append article dict to list
            potential_articles.append({"title": article["title"], "url": url, "description": article["description"], "text": text})

            # append the url to the processed urls
            state["scraped_urls"].append(url)

    # append the processed articles to the state
    state["potential_articles"].extend(potential_articles)

    return state

def select_top_urls(state: GraphState) -> GraphState:
    """Based on the article synopses, choose the top-n articles to summarize."""
    news_query = state["news_query"]
    num_articles_tldr = state["num_articles_tldr"]
    
    # load all processed articles with full text but no summaries
    potential_articles = state["potential_articles"]

    # format the metadata
    formatted_metadata = "\n".join([f"{article['url']}\n{article['description']}\n" for article in potential_articles])

    prompt = f"""
    Based on the user news query: {news_query},
    Reply with a list of strings of up to {num_articles_tldr} relevant urls.
    Don't add any urls that are not relevant or aren't listed specifically.
    {formatted_metadata}
    """
    result = chat_model.invoke(prompt).content

    # use regex to extract the urls as a list
    url_pattern = r'(https?://[^\s",]+)'

    # Find all URLs in the text
    urls = re.findall(url_pattern, result)

    # add the selected article metadata to the state
    tldr_articles = [article for article in potential_articles if article['url'] in urls]
    state["tldr_articles"] = tldr_articles

    return state

async def summarize_articles_parallel(state: GraphState) -> GraphState:
    """Summarize the articles based on full text."""
    tldr_articles = state["tldr_articles"]

    prompt = """
    Create a bulleted summary for the article:
    {text}
      
    Be sure to strictly follow the following format:
    Title: {title} (in bold text)
    Source: {url}
    * bulleted summary
    -------------------------------------
    """

    # iterate over the selected articles and collect summaries synchronously
    for i in range(len(tldr_articles)):
        text = tldr_articles[i]["text"]
        title = tldr_articles[i]["title"]
        url = tldr_articles[i]["url"]
        # invoke the llm synchronously
        result = chat_model.invoke(prompt.format(title=title, url=url, text=text))
        tldr_articles[i]["summary"] = result.content

    state["tldr_articles"] = tldr_articles

    return state

def format_results(state: GraphState) -> GraphState:
    """Format the results for display."""
    # load a list of past search queries
    q = [newsapi_params["q"] for newsapi_params in state["past_searches"]]
    formatted_results = f"Here are the top {len(state['tldr_articles'])} articles based on search terms:\n{', '.join(q)}\n\n"

    # load the summarized articles
    tldr_articles = state["tldr_articles"]

    # format article tl;dr summaries
    tldr_articles = "\n\n".join([f"{article['summary']}" for article in tldr_articles])

    # concatenate summaries to the formatted results
    formatted_results += tldr_articles

    state["formatted_results"] = formatted_results

    return state

## Langgraph Workflow
def articles_text_decision(state: GraphState) -> str:
    """Check results of retrieve_articles_text to determine next step."""
    if state["num_searches_remaining"] == 0:
        # if no articles with text were found return END
        if len(state["potential_articles"]) == 0:
            state["formatted_results"] = "No articles with text found."
            return "END"
        # if some articles were found, move on to selecting the top urls
        else:
            return "select_top_urls"
    else:
        # if the number of articles found is less than the number of articles to summarize, continue searching
        if len(state["potential_articles"]) < state["num_articles_tldr"]:
            return "generate_newsapi_params"
        # otherwise move on to selecting the top urls
        else:
            return "select_top_urls"
        

workflow = Graph()

workflow.add_node("generate_newsapi_params", generate_newsapi_params)
workflow.add_node("retrieve_articles_metadata", retrieve_articles_metadata)
workflow.add_node("retrieve_articles_text", retrieve_articles_text)
workflow.add_node("select_top_urls", select_top_urls)
workflow.add_node("summarize_articles_parallel", summarize_articles_parallel)
workflow.add_node("format_results", format_results)

workflow.add_edge(START, "generate_newsapi_params")
workflow.add_edge("generate_newsapi_params", "retrieve_articles_metadata")
workflow.add_edge("retrieve_articles_metadata", "retrieve_articles_text")
# # if the number of articles with parseable text is less than number requested, then search for more articles
workflow.add_conditional_edges(
    "retrieve_articles_text",
    articles_text_decision,
    {
        "generate_newsapi_params": "generate_newsapi_params",
        "select_top_urls": "select_top_urls",
        "END": END
    }
    )
workflow.add_edge("select_top_urls", "summarize_articles_parallel")
workflow.add_conditional_edges(
    "summarize_articles_parallel",
    lambda state: "format_results" if len(state["tldr_articles"]) > 0 else "END",
    {
        "format_results": "format_results",
        "END": END
    }
    )
workflow.add_edge("format_results", END)
app = workflow.compile()

async def run_workflow(
    query: str,
    num_searches_remaining: int = 5,
    num_articles_tldr: int = 3,
    #sources: Optional[str] = None,
    language: str = "en",
    sort_by: str = "relevancy",
    searchIn: Optional[List[str]] = None
):
    """Run the LangGraph workflow and display results with optional news API parameters."""

    # Dynamically set date range (from 8 days ago to today)
    today = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=8)).strftime("%Y-%m-%d")

    # Construct newsapi_params dynamically
    newsapi_params = {
        "q": query,
        "from_param": from_date,
        "to": today,
        "language": language,
        "sort_by": sort_by,
        "searchIn": searchIn
    }
    
    if searchIn:  # If searchIn is provided
        newsapi_params["searchIn"] = ",".join(searchIn)

    # Construct the initial state with the updated GraphState
    initial_state: GraphState = {
        "news_query": query,
        "num_searches_remaining": num_searches_remaining,
        "newsapi_params": newsapi_params, 
        "past_searches": [],
        "articles_metadata": [],
        "scraped_urls": [],
        "num_articles_tldr": num_articles_tldr,
        "potential_articles": [],
        "tldr_articles": [],
        "formatted_results": "No articles with text found.",
    }

    try:
        result = await app.ainvoke(initial_state)
        return result["formatted_results"]
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        #traceback.print_exc()
        return None
