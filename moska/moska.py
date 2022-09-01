import Deck
from Game import MoskaGame
from Player import MoskaPlayer

if __name__ == "__main__":
    deck = Deck.StandardDeck()
    moskaGame = MoskaGame(deck)
    players = []
    for i in range(3):
        players.append(MoskaPlayer(moskaGame,pid=i))
    for player in players:
        moskaGame.add_player(player)
    moskaGame.start()

exit()


from collections import deque
from dataclasses import dataclass
import itertools as it
import random
from typing import Iterable


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
        get_valid_deck(deck).place_to_bottom(self.triumph_card)
    
    def __repr__(self):
        s = f"Triumph card: {self.triumph_card}\n"
        for pl in self.players:
            s += f"#{pl.pid}{'*' if self.turn_pid == pl.pid else ''} : {pl.hand}\n"
        s += f"Cards to fall : {self.cards_to_fall}\n"
        s += f"Fell cards : {self.fell_cards}\n"
        return s
    
    def get_players(self,cond = lambda x : True):
        return list(filter(cond,self.players))
    
    def add_cards_to_fall(self,add):
        self.cards_to_fall += add
    
    def fall_card(self,played_card, fall_card):
        success = False
        # Jos kortit ovat samaa maata ja pelattu kortti on suurempi
        if played_card.suit == fall_card.suit and played_card.value > fall_card.value:
            success = True
        # Jos pelattu kortti on valttia, ja kaadettava kortti ei ole valttia
        elif played_card.suit == self.triumph and fall_card.suit != self.triumph:
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
    
    def get_active_player(self):
        return moskaTable.get_players(cond = lambda x : x.pid == self.moskaTable.turn_pid)[0]     # Get the player to whom the cards are played to
    
    def next_turn_automatic(self,initial=False):
        print(self.moskaTable)
        turn_pid = self.moskaTable.turn_pid
        player = self.get_active_player()
        if player.rank is not None:
            self._end_turn()
            return
        # If no cards have been played to player in turn
        if initial:
            prev_player = player.get_prev_player()  # The previous player gets to make a free first move
            prev_player.play_to_another_automatic(initial=True,fits=len(player.hand))
            print(moskaTable)
        other_players = lambda : moskaTable.get_players(cond = lambda x : x.pid != turn_pid and x.rank is None)
        players_left = lambda : len(self.moskaTable.get_players(cond=lambda x : x.rank is None))
        played = True
        # The other players play cards to the target
        while played:
            played = False
            # If there is only one person with cards, he lost
            if players_left() <= 1:
                self.moskaTable.get_players(cond = lambda x : x.rank is None)[0].set_rank()
                break
            # If the table is full
            fits = len(player.hand) - len(self.moskaTable.cards_to_fall)
            if fits <= 0:
                break
            for ot_pl in other_players():
                res = ot_pl.play_to_another_automatic(fits=fits)
                played = played if played else res
            print(self.moskaTable)
        played = player.fell_cards_automatic()
        print(self.moskaTable)
        if played:
            self.next_turn_automatic()
        elif initial:
            self._end_turn()
            
        
    def _end_turn(self):
        player = self.get_active_player()
        lifted = False
        if self.moskaTable.cards_to_fall:
            lifted = True
            player.hand.add(self.moskaTable.cards_to_fall)
        self.moskaTable.cards_to_fall = []
        self.moskaTable.fell_cards = []
        if not lifted:
            self.moskaTable.turn_pid = player.get_next_player().pid
        else:
            self.moskaTable.turn_pid = player.get_next_player().get_next_player().pid
        player.hand.draw(6-len(player.hand))
        
        
class MoskaPlayer:
    hand = None
    pid = 0
    moskaTable = None
    rank = None
    def __init__(self,table : MoskaTable, pid : int = 0):
        self.moskaTable = table
        self.hand = MoskaHand()
        self.pid = pid
        
    def set_rank(self):
        if self.rank is None and not self.hand and len(get_valid_deck()) == 0:
            self.rank = len(self.moskaTable.get_players(cond = lambda x : x.rank is not None)) + 1
        return self.rank
        
        
    def fold(self,deck=None):
        deck = get_valid_deck(deck)
        deck.add(self.hand.pop_cards())
    
    def play_to_another_automatic(self,fits = 6,initial = False):
        play = []
        if not self.hand:
            return False
        min_card = min([c.value for c in self.hand])
        if initial:
            play = self.hand.pop_cards(cond=lambda x : x.value == min_card,max_cards = fits)
        else:
            playable_values = set([c.value for c in self.moskaTable.cards_to_fall + self.moskaTable.fell_cards]).intersection([c.value for c in self.hand])
            if playable_values:
                play = self.hand.pop_cards(cond=lambda x : x.value in playable_values,max_cards = fits)
        moskaTable.add_cards_to_fall(play)
        self.hand.cards += get_valid_deck().pop_cards(len(play))
        self.set_rank()
        return bool(play)
    
    def fell_cards_automatic(self):
        play_cards = {}
        played = False
        for fall_card in moskaTable.cards_to_fall:
            cards_that_fall = []
            for play_card in self.hand:
                success = moskaTable.fall_card(play_card,fall_card)
                if success:
                    cards_that_fall.append(play_card)
            if cards_that_fall:
                play_cards[min(cards_that_fall)] = fall_card
        for played, fell in play_cards.items():
            # Add the fell card and the played card to the list of fallen cards
            moskaTable.fell_cards.append(moskaTable.cards_to_fall.pop(moskaTable.cards_to_fall.index(fell)))
            moskaTable.fell_cards.append(self.hand.pop_cards(cond = lambda x : x == played)[0])
            played = True
        self.set_rank()
        return played
        
    
        
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
    while len(moskaTable.get_players(cond = lambda x : x.rank is None)) > 1:
        print("Cards in deck: ", len(MAIN_DECK),"\n\n\n\n")
        moskaGame.next_turn_automatic(initial=True)
    
    