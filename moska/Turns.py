from typing import Callable
from Game import MoskaGame
from Player import MoskaPlayer
import utils

class _PlayToPlayer:
    """ This is the class of plays, that players can make, when they play cards to someone else or to themselves."""
    def __init__(self,moskaGame : MoskaGame, player : MoskaPlayer):
        """
        Args:
            moskaGame (MoskaGame): _description_
            player (MoskaPlayer): _description_
            target_player (MoskaPlayer): _description_
            play_cards (list): _description_
        """
        self.moskaGame = moskaGame
        self.player = player
        
    def check_cards_available(self):
        """ Check that the cards are playable"""
        return all([card in self.player.hand.cards for card in self.play_cards])
    
    def check_fits(self):
        """ Check that the cards fit in the table """
        return len(self.target_player.hand) - len(self.moskaGame.cards_to_fall) >= len(self.play_cards)
    
    def play(self):
        self.player.hand.pop_cards(lambda x : x in self.play_cards) # Remove the played cards from the players hand
        self.moskaGame.add_cards_to_fall(self.play_cards)           # Add the cards to the cards_to_fall -list
        self.player.hand.draw(len(self.play_cards))                 # Draw the amount of cards played, from the deck
        self.player.set_rank()
        

class InitialPlay(_PlayToPlayer):
    """ The play that must be done, when it is the players turn to play cards to a target (and only then)"""
    def __call__(self,target_player : MoskaPlayer, play_cards : list):
        self.target_player = target_player
        self.play_cards = play_cards
        assert self.check_cards_available(), "Some of the played cards are not available"
        assert self.check_fits(), "Attempted to play too many cards."
        self.play()
        
class PlayToOther(_PlayToPlayer):
    """ This is the play, that players can constantly make when playing cards to an opponent after the initial play"""
    def __call__(self, target_player : MoskaPlayer, play_cards : list):
        #self.__init__(moskaGame,player,target_player,play_cards)
        self.target_player = target_player
        self.play_cards = play_cards
        assert self.check_cards_available(), "Some of the played cards are not available"
        assert self.check_fits(), "Attempted to play too many cards."
        assert self.check_in_table(), "Some of the cards you tried to play, are not playable, because they haven't yet been played by another player."
        self.play()
    
    def _playable_values(self):
        """ Return a set of values, that can be played to target"""
        return set([c.value for c in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards])
    
    def check_in_table(self):
        """ Check that the cards have already been played by either the player or an opponent"""
        playable_values = self._playable_values()
        return all([card.value in playable_values for card in self.play_cards])


class PlayFallCardFromHand:
    """ A class, that is used when playing cards from the hand, to fall cards on the table"""
    def __init__(self,moskaGame : MoskaGame, player : MoskaPlayer):
        self.moskaGame = moskaGame
        self.player = player

    def __call__(self,play_fall : dict):
        self.play_fall = play_fall
        assert self.check_cards_fall(), "Some of the played cards were not matched to a correct card to fall."
        self.play()
        
    def check_cards_fall(self):
        """Returns whether all the pairs are correctly played"""
        return all([utils.check_can_fall_card(pc,fc,self.moskaGame.triumph) for pc,fc in self.play_fall.items()])
        
    def play(self):
        for pc,fc in self.play_fall.items():
            self.moskaGame.cards_to_fall.pop(self.moskaGame.cards_to_fall.index(fc))        # Remove from cards_to_fall
            self.moskaGame.fell_cards.append(fc)                                            # Add to fell cards
            self.player.hand.pop_cards(cond = lambda x : x == pc)                           # Remove card from hand
            self.moskaGame.fell_cards.append(pc)                                            # Add card to fell cards
            self.player.set_rank()
        
            
class PlayFallFromDeck:
    def __init__(self,moskaGame : MoskaGame,fall_method : Callable = None):
        self.moskaGame = moskaGame
        self.fall_method = fall_method
        
    def __call__(self, fall_method : Callable = None):
        """ fall_method must accept one argument: the card that was drawn,
        and return an indexable with two values: The played card, and the card which should be fallen"""
        if fall_method is not None:
            self.fall_method = fall_method
        assert self.fall_method is not None, "No fall_method specified"
        assert self.check_cards_on_table(), "There are no cards on the table which should be fell"
        assert self.check_cards_in_deck(), "There are no cards from which to draw"
        self.play_card()
    
    def check_cards_on_table(self):
        """ Check if there are un-fallen cards on the table"""
        return bool(self.moskaGame.cards_to_fall)
    
    def check_cards_in_deck(self):
        """ Check that there are cards on the table, from which to draw"""
        return len(self.moskaGame.deck) > 0
    
    def play_card(self):
        self.card = self.moskaGame.deck.pop_cards(1)[0]
        if self.check_can_fall():
            play_fall = self.fall_method(self.card)
            self.moskaGame.cards_to_fall.pop(self.moskaGame.cards_to_fall.index(play_fall[1]))
            self.moskaGame.fell_cards.append(play_fall[1])
            self.moskaGame.fell_cards.append(play_fall[0])
        else:
            self.moskaGame.add_cards_to_fall([self.card])
            
    def check_can_fall(self):
        return any([utils.check_can_fall_card(self.card,fc,self.moskaGame.triumph) for fc in self.moskaGame.cards_to_fall])

class EndTurn:
    def __init__(self,moskaGame : MoskaGame, player : MoskaPlayer):
        self.moskaGame = moskaGame
        self.player = player
    
    
    def __call__(self,pick_cards : list = []):
        self.pick_cards = pick_cards
        assert self.check_has_played_cards(), "There are no played cards, and hence the turn cannot be ended yet."
        assert self.check_turn(), "It is not this players turn to lift the cards"
        if not pick_cards:
            assert self.check_can_pick_none(), "There are cards on the table, and they must fall or be lifted."
        else:
            assert self.check_pick_all_cards() or self.check_pick_cards_to_fall()
        self.pick_the_cards()
    
    def clear_table(self):
        self.moskaGame.cards_to_fall = []
        self.moskaGame.fell_cards = []
        
    def check_can_pick_none(self):
        """ Check if there are cards to fall"""
        return len(self.moskaGame.cards_to_fall) == 0
    
    def check_turn(self):
        return self.moskaGame.get_active_player() == self.player
    
    def check_pick_cards_to_fall(self):
        """ Check if every pick_card equals cards_to_fall"""
        #return all([card in self.moskaGame.cards_to_fall for card in self.pick_cards])
        return all([picked == card for picked,card in zip(self.pick_cards,self.moskaGame.cards_to_fall)])
    
    def check_pick_all_cards(self):
        """ Check if every pick_card is in either cards_to_fall or in fell_cards and not every card is in cards_to_fall"""
        #return all([card in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards for card in self.pick_cards]) and not self.check_pick_cards_to_fall()
        return all([picked == card for picked,card in zip(self.pick_cards,self.moskaGame.cards_to_fall + self.moskaGame.fell_cards)])
    
    def check_has_played_cards(self):
        """ Check if there are cards that are not fallen or that are fallen"""
        return bool(self.moskaGame.cards_to_fall + self.moskaGame.fell_cards)
    
    def pick_the_cards(self):
        self.player.hand.cards += self.pick_cards
        self.player.hand.draw(6 - len(self.player.hand))
        self.clear_table()
        self.player.set_rank()
    
    
    
    
    
        