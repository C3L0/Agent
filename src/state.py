import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # 'messages' est la liste de tous les échanges. 
    # 'Annotated[..., operator.add]' permet d'ajouter les nouveaux messages à la liste existante
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # On peut ajouter des champs personnalisés pour structurer les données
    next_agent: str  # Pour le routing
    extracted_data: list[str]  # Pour stocker des résultats de recherche intermédiaires
