from copy import copy, deepcopy
from Game import MoskaGame
from Hand import MoskaHand
import utils
#import Turns
#from Turns import PlayFallCardFromHand, PlayFallFromDeck, PlayToOther, InitialPlay, EndTurn

class MoskaPlayer:
    hand = None
    pid = 0
    moskaGame = None
    rank = None
    thread = None
    def __init__(self,moskaGame : MoskaGame, pid : int = 0):
        self.moskaGame = moskaGame
        self.hand = MoskaHand(moskaGame)
        self.pid = pid
        from Turns import PlayFallCardFromHand, PlayFallFromDeck, PlayToOther, InitialPlay, EndTurn
        self.playFallCardFromHand = PlayFallCardFromHand(self.moskaGame,self)
        self.playFallFromDeck = PlayFallFromDeck(self.moskaGame)
        self.playToOther = PlayToOther(self.moskaGame,self)
        self.initialPlay = InitialPlay(self.moskaGame,self)
        self.endTurn = EndTurn(self.moskaGame,self)
        
    def set_pid(self,pid):
        self.pid = pid
    
    def _playable_values(self):
        """ Return a set of values, that can be played to target"""
        return set([c.value for c in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards]).intersection([c.value for c in self.hand])
    
    def play_to_target(self):
        """ Return a list of cards that will be played to the target player """
        target = self.moskaGame.get_active_player()
        playable_values = self._playable_values()
        play_cards = []
        if playable_values:
            hand = deepcopy(self.hand)
            play_cards = hand.pop_cards(cond=lambda x : x.value in playable_values,max_cards = len(target.hand) - len(self.moskaGame.cards_to_fall))
        self.playToOther(target,play_cards)
        
    def play_initial(self):
        target = self.moskaGame.get_active_player()
        min_card = min([c.value for c in self.hand])
        #play_cards = []
        hand = deepcopy(self.hand)
        play_cards = hand.pop_cards(cond=lambda x : x.value == min_card,max_cards = len(target.hand) - len(self.moskaGame.cards_to_fall))
        self.initialPlay(target,play_cards)
    
    def play_fall_from_deck(self):
        def fall_method(deck_card):
            for card in self.moskaGame.cards_to_fall:
                if utils.check_can_fall_card(deck_card,card):
                    return (deck_card,card)
        self.playFallFromDeck(fall_method=fall_method)
    
    def play_fall_card_from_hand(self):
        play_cards = {}
        for fall_card in self.moskaGame.cards_to_fall:
            cards_that_fall = []
            for play_card in self.hand:
                success = utils.check_can_fall_card(play_card,fall_card,self.moskaGame.triumph)
                if success:
                    cards_that_fall.append(play_card)
            if cards_that_fall:
                play_cards[min(cards_that_fall)] = fall_card
        self.playFallCardFromHand(play_cards)
        
    def end_turn(self):
        pick_cards = self.moskaGame.cards_to_fall
        self.endTurn(pick_cards)
        
    def set_rank(self):
        if self.rank is None and not self.hand and len(self.moskaGame.deck) == 0:
            self.rank = len(self.moskaGame.get_players_condition(cond = lambda x : x.rank is not None)) + 1
        return self.rank
        
    def fold(self):
        self.moskaGame.deck.add(self.hand.pop_cards())
        
"""

class MoskaPlayerBot(MoskaPlayer):
    
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
"""