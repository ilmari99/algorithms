from collections import deque
from dataclasses import dataclass
import itertools as it
import random
from typing import Iterable

CARD_VALUES = tuple(range(2,15))
CARD_SUITS = ("C","D","H","S")
CARD_SUIT_SYMBOLS = {"S":'♠', "D":'♦',"H": '♥',"C": '♣'}
MAIN_DECK = None

def suit_to_symbol(suits):
    if isinstance(suits,str):
        return CARD_SUIT_SYMBOLS[suits]
    return [CARD_SUIT_SYMBOLS[s] for s in suits]

def get_valid_deck(deck=None):
    deck = deck if deck else MAIN_DECK
    if deck is None:
        raise AssertionError("The deck from which to draw is not specified!")
    return deck

@dataclass
class Card:
    value : int
    suit : str
    
    def __repr__(self) -> str:
        return str(f"{suit_to_symbol(self.suit)}{self.value}")
    
    def as_str(self,symbol=True):
        if symbol:
            return self.__repr__()
        return str(f"{self.suit}{self.value}")


# Could probabably be converted to a subclass of deque
class StandardDeck:
    def __init__(self,shuffle=True):
        self.cards = deque((Card(v,s) for v,s in it.product(CARD_VALUES,CARD_SUITS)))
        if shuffle:
            self.shuffle()
        return None
    
    def __len__(self):
        return len(self.cards)
    
    def shuffle(self):
        random.shuffle(self.cards)
        return None
    
    def pop_cards(self,n):
        if not self.cards:
            return []
        return [self.cards.pop() for _ in range(min(len(self),n))]
    
    def add(self,cards : Iterable[Card]):
        for card in cards:
            if not isinstance(card,Card):
                raise TypeError("The iterable to add to the deck, must be of type Card")
            self.cards.append(card)
        self.shuffle()
        return None
    
    def insert(self,pos,card):
        self.cards.insert(pos,card)
        
        

class MoskaHand:
    cards = []
    def __init__(self,deck=None):
        deck = get_valid_deck(deck)
        self.cards = deck.pop_cards(6)
    
    def draw(self,n,deck = None):
        deck = get_valid_deck(deck)
        self.cards += deck.pop_cards(n)
        return None
    
    def add(self,cards):
        self.cards += cards
        
    
    def pop_card(self,card=None):
        return self.cards.pop(self.cards.index(card))
    
    def pop_cards(self,cond = lambda x : True):
        return [self.cards.pop(self.cards.index(card)) for card in self.cards.copy() if cond(card)]
    
    def __repr__(self) -> str:
        s = ""
        s = s.join((c.as_str()+" " for c in self.cards))
        return s
    
    
    def __iter__(self):
        return iter(self.cards)
    
    def __len__(self):
        return len(self.cards)
    

class MoskaTable:
    players = []
    triumph = ""
    triumph_card = None
    cards_to_fall = []
    fell_cards = []
    turn_pid = 0
    def __init__(self, nplayers : int, deck = None, turn_pid=0):
        assert nplayers > 1 and nplayers <= 8, "Too few, or too many players"
        self.players = [MoskaPlayer(self,pid=pid) for pid in range(nplayers)]
        self.triumph_card = get_valid_deck(deck).pop_cards(1)[0]
        self.triumph = self.triumph_card.suit
        get_valid_deck(deck).insert(-1,self.triumph_card)
    
    def __repr__(self):
        s = f"Triumph card: {self.triumph_card}\n"
        for pl in self.players:
            s += f"#{pl.pid}{'*' if self.turn_pid == pl.pid else ''} : {pl.hand}\n"
        s += f"Cards to fall : {self.cards_to_fall}\n"
        return s
    
    def get_players(self,cond = lambda x : True):
        return list(filter(cond,self.players))
    
    def add_cards_to_fall(self,add):
        self.cards_to_fall += add
    
    def fall_card(self,played_card, fall_card):
        success = False
        if played_card.suit == self.triumph:
            if fall_card.suit != self.triumph or fall_card.value < played_card.value:
                success = True
        elif fall_card.value < played_card.value:
            success = True
        return success
        #if not success:
        #    raise AssertionError("The played card {} does not fall card {}".format(played_card.as_str(),fall_card.as_str()))    
        #self.fell_cards += [played_card,fall_card]
    
    def updatePids(self):
        for i,pl in enumerate(self.players):
            pl.pid = i
    
            
        

class MoskaGame:
    def __init__(self,moskaTable : MoskaTable):
        self.moskaTable = moskaTable
    
    def remove_players(self,cond = lambda x : len(x.hand) < 1):
        self.moskaTable.players = list(filter(cond,self.moskaTable.players))
        
    def next_turn_automatic(self):
        print(moskaTable)
        turn_pid = moskaTable.turn_pid
        player = moskaTable.get_players(cond = lambda x : x.pid == turn_pid)[0]
        state_of_table = moskaTable.cards_to_fall.copy()    # Cards to fall
        # If no cards have been played to player in turn
        if not self.moskaTable.cards_to_fall and not self.moskaTable.fell_cards:
            prev_player = player.get_prev_player()  # The previous player gets to make a free first move
            prev_player.play_to_another_automatic(initial=True)
            print(moskaTable)
        other_players = moskaTable.get_players(cond = lambda x : x.pid != turn_pid)
        plays = 0
        # The players may add cards for as long as there is room on the table
        # If there is a change in the table, check if other players can play
        while plays < 1 and len(moskaTable.cards_to_fall) < len(player.hand) and state_of_table != moskaTable.cards_to_fall:
            state_of_table = moskaTable.cards_to_fall.copy()
            for ot_pl in other_players:
                ot_pl.play_to_another_automatic()
            print(moskaTable)
        #self.remove_players(cond = lambda x : len(x.hand) > 0 and x.pid != turn_pid)
        player.fell_cards_automatic()
        print(moskaTable)
        if state_of_table != moskaTable.cards_to_fall:
            self.next_turn_automatic()
        if moskaTable.cards_to_fall:
            player.hand.add(moskaTable.cards_to_fall)
        moskaTable.cards_to_fall = []
        moskaTable.fell_cards = []
        moskaTable.turn_pid = player.get_next_player().pid
        player.hand.draw(6-len(player.hand))
        moskaTable.updatePids()
        
        
        
            
        
                
    
    
class MoskaPlayer:
    hand = None
    pid = 0
    moskaTable = None
    def __init__(self,table : MoskaTable, pid : int =0):
        self.moskaTable = table
        self.hand = MoskaHand()
        self.pid = pid
        
    def fold(self,deck=None):
        deck = get_valid_deck(deck)
        deck.add(self.hand.pop_cards())
    
    def play_to_another_automatic(self,initial = False):
        play = []
        min_card = min([c.value for c in self.hand])
        if initial:
            play = self.hand.pop_cards(cond=lambda x : x.value == min_card)
        else:
            playable_values = set([c.value for c in self.moskaTable.cards_to_fall + self.moskaTable.fell_cards]).intersection([c.value for c in self.hand])
            if playable_values:
                play = self.hand.pop_cards(cond=lambda x : x.value in playable_values)
        moskaTable.add_cards_to_fall(play)
        self.hand.cards += get_valid_deck().pop_cards(len(play))
    
    def fell_cards_automatic(self):
        for fall_card in moskaTable.cards_to_fall:
            for play_card in self.hand:
                success = moskaTable.fall_card(play_card,fall_card)
                if success:
                    moskaTable.cards_to_fall.remove(fall_card)
                    #moskaTable.cards_to_fall.pop(moskaTable.cards_to_fall.index(fall_card))
                    self.hand.pop_cards(cond = lambda x : x == play_card)
                    break
                
        
    def get_prev_player(self):
        if self.pid == 0:
            prev_pid = len(moskaTable.players)-1
        else:
            prev_pid = self.pid - 1
        return moskaTable.get_players(cond = lambda x : x.pid == prev_pid)[0]
    
    def get_next_player(self):
        if self.pid == len(moskaTable.players)-1:
            next_pid = 0
        else:
            next_pid = self.pid + 1
        return moskaTable.get_players(cond = lambda x : x.pid == next_pid)[0]
    
    
        
    
    



if __name__ == "__main__":
    MAIN_DECK = StandardDeck()
    print(len(MAIN_DECK))
    moskaTable = MoskaTable(3,MAIN_DECK)
    moskaGame = MoskaGame(moskaTable)
    while moskaTable.players:
        print("Cards in deck: ", len(MAIN_DECK),"\n\n\n\n")
        moskaGame.next_turn_automatic()
    
    