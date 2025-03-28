import asyncio
import aiohttp
import feedparser
from bs4 import BeautifulSoup
import requests
from difflib import SequenceMatcher
from datetime import datetime
from collections import defaultdict
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(self):
        self.memory = defaultdict(dict)
        self.preferences = {}

    def store_interaction(self, query, cycle, data):
        self.memory[query.lower()][cycle] = data

    def recall(self, query, cycle=None):
        query = query.lower()
        if cycle:
            return self.memory.get(query, {}).get(cycle)
        return self.memory.get(query)

class WebExtractor:
    async def fetch_arxiv(self, query):
        """Try arXiv first"""
        base_url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 3,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        headers = {'User-Agent': 'AcademicResearchBot/1.0'}
        
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(base_url, params=params, timeout=20) as response:
                    if response.status == 200:
                        feed = feedparser.parse(await response.text())
                        papers = []
                        for entry in feed.entries[:3]:
                            try:
                                published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
                                papers.append({
                                    "title": entry.title,
                                    "abstract": entry.summary,
                                    "published": published.strftime("%d %b %Y"),
                                    "authors": [a.name for a in entry.authors],
                                    "link": entry.link,
                                    "pdf_link": entry.link.replace("abs", "pdf") + ".pdf",
                                    "source": "arXiv"
                                })
                            except Exception as e:
                                logger.warning(f"Error parsing arXiv entry: {e}")
                        return papers
        except Exception as e:
            logger.error(f"arXiv API error: {e}")
            return []

    async def fetch_ieee(self, query):
        """Try IEEE if arXiv fails"""
        url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            papers = []
            
            for result in soup.select('.List-results-items')[:3]:
                try:
                    title = result.select_one('.h3 a').text.strip()
                    authors = [a.text.strip() for a in result.select('.publisher-info-container a')]
                    abstract = result.select_one('.description') and result.select_one('.description').text.strip()
                    link = 'https://ieeexplore.ieee.org' + result.select_one('.list-article-title a')['href']
                    
                    papers.append({
                        "title": title,
                        "authors": authors,
                        "abstract": abstract or "Abstract not available",
                        "link": link,
                        "source": "IEEE"
                    })
                except Exception as e:
                    logger.warning(f"Error parsing IEEE result: {e}")
            return papers
        except Exception as e:
            logger.error(f"IEEE scraping error: {e}")
            return []

    async def fetch_semantic_scholar(self, query):
        """Try Semantic Scholar if IEEE fails"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": 3,
            "fields": "title,authors,abstract,url,year,openAccessPdf"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        papers = []
                        for paper in data.get('data', []):
                            papers.append({
                                "title": paper.get('title', ''),
                                "authors": [a['name'] for a in paper.get('authors', [])],
                                "abstract": paper.get('abstract', 'Abstract not available'),
                                "published": str(paper.get('year', '')),
                                "link": paper.get('url', '#'),
                                "pdf_link": paper.get('openAccessPdf', {}).get('url', ''),
                                "source": "Semantic Scholar"
                            })
                        return papers
        except Exception as e:
            logger.error(f"Semantic Scholar API error: {e}")
            return []

    async def fetch_web_search(self, query):
        """Final fallback to web search"""
        API_KEY = os.getenv('SERPAPI_KEY')
        if not API_KEY:
            logger.warning("SerpAPI key not found")
            return []
            
        url = f"https://serpapi.com/search.json?q={query}+filetype:pdf&api_key={API_KEY}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        papers = []
                        for result in data.get('organic_results', [])[:3]:
                            if 'pdf' in result.get('link', ''):
                                papers.append({
                                    "title": result.get('title', ''),
                                    "link": result.get('link', '#'),
                                    "pdf_link": result.get('link', '#'),
                                    "source": "Web Search",
                                    "abstract": result.get('snippet', 'Abstract not available'),
                                    "published": ""
                                })
                        return papers
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

class ResearchAgent:
    def __init__(self, memory):
        self.memory = memory

    async def generate(self, query, cycle):
        related = self.memory.recall(query.lower())
        if related:
            return f"Building on previous research about {query}"
        return f"Initial hypothesis about {query}"

class ReflectionAgent:
    async def validate(self, papers):
        if not papers:
            return "⚠️ No supporting research found", 0
        
        valid_sources = len([p for p in papers if p.get('title') and p.get('abstract')])
        coherence_score = min(10, valid_sources * 2)
        return f"✅ Validated with {valid_sources} sources", coherence_score

class RankingAgent:
    async def evaluate(self, query, papers):
        if not papers:
            return {"score": 0, "top_paper": {}, "all_papers": []}

        scored_papers = []
        current_date = datetime.now()

        for paper in papers:
            try:
                title_score = SequenceMatcher(None, query.lower(), paper["title"].lower()).ratio()
                abstract_score = SequenceMatcher(None, query.lower(), paper.get("abstract", "").lower()).ratio()
                
                recency_score = 0.5
                if paper.get("published"):
                    try:
                        if paper["source"] == "arXiv":
                            pub_date = datetime.strptime(paper["published"], "%d %b %Y")
                        else:
                            pub_date = datetime.strptime(paper["published"], "%Y") if len(paper["published"]) == 4 else datetime.strptime(paper["published"], "%d %b %Y")
                        days_old = (current_date - pub_date).days
                        recency_score = max(0, 1 - days_old / 730)
                    except:
                        pass
                
                source_weights = {
                    "arXiv": 1.2,
                    "IEEE": 1.1,
                    "Semantic Scholar": 1.1,
                    "Web Search": 1.0
                }
                source_weight = source_weights.get(paper.get("source", "Web Search"), 1.0)
                
                pdf_bonus = 0.1 if paper.get("pdf_link") else 0
                
                composite = 10 * (0.4 * title_score + 0.3 * abstract_score + 0.2 * recency_score + 0.1 * source_weight) + pdf_bonus
                
                paper["score"] = round(composite, 1)
                scored_papers.append(paper)
            except Exception as e:
                logger.warning(f"Error scoring paper {paper.get('title')}: {e}")

        scored_papers.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "score": scored_papers[0]["score"] if scored_papers else 0,
            "top_paper": scored_papers[0] if scored_papers else {},
            "all_papers": scored_papers
        }

class EvolutionAgent:
    def __init__(self, memory):
        self.memory = memory

    async def refine(self, hypothesis, papers, query):
        if not papers:
            return hypothesis
        
        top_keywords = []
        for paper in papers[:3]: 
            if 'title' in paper:
                top_keywords.extend(paper['title'].split()[:5])
        
        if top_keywords:
            enhanced = f"{hypothesis} (enhanced with: {', '.join(set(top_keywords[:3]))})"
            return enhanced
        return hypothesis

class ProximityAgent:
    def __init__(self, memory):
        self.memory = memory

    async def find_related(self, query):
        query = query.lower()
        past_queries = list(self.memory.memory.keys())
        if not past_queries:
            return None
            
        similar_queries = []
        for past_query in past_queries:
            similarity = SequenceMatcher(None, query, past_query.lower()).ratio()
            if similarity > 0.4:  
                best_data = self.memory.recall(past_query)
                if best_data:
                    best_cycle = max(best_data.keys())
                    similar_queries.append({
                        'query': past_query,
                        'score': best_data[best_cycle]['score'],
                        'hypothesis': best_data[best_cycle]['hypothesis']
                    })
        
        if similar_queries:
            similar_queries.sort(key=lambda x: x['score'], reverse=True)
            return [f"Related to '{item['query']}' (score: {item['score']}/10): {item['hypothesis']}" 
                   for item in similar_queries[:3]]
        return None

class MetaReviewAgent:
    def __init__(self, memory):
        self.memory = memory
        self.performance_log = []

    async def review_process(self, query, cycle_time, sources_used):
        feedback = []
        
        if not sources_used:
            feedback.append("⚠️ All sources failed - consider expanding search parameters")
        elif "Web Search" in sources_used and len(sources_used) == 1:
            feedback.append("⚠️ Only web search results available - academic sources failed")
        
        if cycle_time > 10:
            feedback.append("⚡ Optimize slow queries with caching")
            
        if len(feedback) == 0:
            feedback.append("✅ Process efficient")
            
        self.performance_log.append({
            "query": query,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        return feedback

class ResearchSystem:
    def __init__(self):
        self.memory = MemorySystem()
        self.web_extractor = WebExtractor()
        self.agents = {
            "generation": ResearchAgent(self.memory),
            "reflection": ReflectionAgent(),
            "ranking": RankingAgent(),
            "evolution": EvolutionAgent(self.memory),
            "proximity": ProximityAgent(self.memory),
            "meta_review": MetaReviewAgent(self.memory)
        }

    async def process_query(self, query, cycles=3):
        query = query.lower()
        state = {
            "query": query,
            "cycle": 0,
            "papers": [],
            "hypothesis": "",
            "score": 0,
            "feedback": [],
            "insights": [],
            "related": None,
            "related_papers": [],
            "agent_status": {
                "generation": "⏳ Waiting...",
                "reflection": "⏳ Waiting...",
                "ranking": "⏳ Waiting...",
                "evolution": "⏳ Waiting...",
                "proximity": "⏳ Waiting...",
                "meta_review": "⏳ Waiting..."
            },
            "meta_feedback": [],
            "sources_used": []
        }

        for cycle in range(1, cycles + 1):
            start_time = datetime.now()
            state["cycle"] = cycle
            state["sources_used"] = []

            # Hypothesis generation
            state["agent_status"]["generation"] = "🔄 Generating hypothesis..."
            state["hypothesis"] = await self.agents["generation"].generate(query, cycle)
            state["agent_status"]["generation"] = "✅ Hypothesis generated"

            # Related research
            state["agent_status"]["proximity"] = "🔄 Finding related research..."
            state["related"] = await self.agents["proximity"].find_related(query)
            state["agent_status"]["proximity"] = "✅ Related research found" if state["related"] else "✅ No related research found"

            # Cascading source check
            state["agent_status"]["reflection"] = "🔄 Gathering research..."
            
            sources = [
                ("arXiv", self.web_extractor.fetch_arxiv),
                ("IEEE", self.web_extractor.fetch_ieee),
                ("Semantic Scholar", self.web_extractor.fetch_semantic_scholar),
                ("Web Search", self.web_extractor.fetch_web_search)
            ]
            
            state["papers"] = []
            for source_name, fetch_method in sources:
                if not state["papers"]:  # Only try next source if no papers found yet
                    papers = await fetch_method(query)
                    if papers:
                        state["papers"] = papers
                        state["sources_used"].append(source_name)
                        logger.info(f"Found {len(papers)} papers from {source_name}")
            
            state["agent_status"]["reflection"] = f"✅ Research gathered from: {', '.join(state['sources_used']) or 'no sources'}"

            # Validation
            validation, coherence = await self.agents["reflection"].validate(state["papers"])
            state["feedback"].append(validation)

            # Hypothesis refinement
            state["agent_status"]["evolution"] = "🔄 Refining hypothesis..."
            state["hypothesis"] = await self.agents["evolution"].refine(state["hypothesis"], state["papers"], query)
            state["agent_status"]["evolution"] = "✅ Hypothesis refined"

            # Ranking
            state["agent_status"]["ranking"] = "🔄 Evaluating papers..."
            ranking = await self.agents["ranking"].evaluate(query, state["papers"])
            state["score"] = ranking["score"]
            state["top_paper"] = ranking["top_paper"]
            state["all_papers"] = ranking["all_papers"]
            state["related_papers"] = ranking["all_papers"][1:4]
            state["agent_status"]["ranking"] = f"✅ Top paper: {state['score']}/10"

            # Meta review
            cycle_time = (datetime.now() - start_time).total_seconds()
            state["agent_status"]["meta_review"] = "🔄 Reviewing process..."
            state["meta_feedback"] = await self.agents["meta_review"].review_process(query, cycle_time, state["sources_used"])
            state["agent_status"]["meta_review"] = "✅ Process reviewed"

            # Store results
            self.memory.store_interaction(query, cycle, {
                "hypothesis": state["hypothesis"],
                "score": state["score"],
                "top_paper": state["top_paper"],
                "related": state["related"],
                "related_papers": state["related_papers"],
                "feedback": state["meta_feedback"],
                "sources": state["sources_used"]
            })

            state["insights"].append(f"🔄 Cycle {cycle}: {state['hypothesis']} (Score: {state['score']}/10)")

        return state