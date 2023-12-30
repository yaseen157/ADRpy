"""Unit tests for the propulsion module."""
import unittest

import numpy as np

from ADRpy.propulsion import EngineDeck


class SeaLevelLapse(unittest.TestCase):

    def test_drythrust(self):
        genericdecktypes = [
            "class:TurbofanHiBPR",
            "class:TurbofanLoBPR",
            "class:Turbojet",
            "class:Turboprop",
            "class:Piston",
            "class:SuperchargedPiston",
            "class:ElectricMotor"
        ]
        for decktype in genericdecktypes:
            deck = EngineDeck(decktype)

            if deck.type in ["piston", "electricmotor"]:
                self.assertTrue(np.isclose(
                    deck.shaftpower(0, 0, norm=True), 1.0, atol=1e-3))
            elif deck.type == "turboprop":
                self.assertTrue(np.isclose(
                    deck.thrust(0, 0, norm=True), 1.0, atol=1e-3))
            else:
                self.assertTrue(np.isclose(
                    deck.thrust(0, 0, norm=True), 1.0, atol=1e-3))

        return


if __name__ == '__main__':
    unittest.main()
