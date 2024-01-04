"""Unit tests for the propulsion module."""
import unittest

import numpy as np

from ADRpy.propulsion import EngineDeck, engine_catalogue


class GenericDecks(unittest.TestCase):
    """Tests for generic, non-dimensionalised engines."""

    def test_lapse_sealevelstatic(self):
        """Test the sea-level static thrust lapse methods. Should ~1 at SLS."""
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


class RealEngineDecks(unittest.TestCase):
    """Test that real engines are able to load their respective deck data."""

    def test_thrust_interpolators(self):
        """The data that was interpolated from, should be reproducible."""
        for (enginename, enginedata) in engine_catalogue.items():
            # Load engine data
            enginedeck = EngineDeck(enginename)

            # Load the correct dataframe
            if "Thrust" in enginedata["dataframes"]:
                df = enginedata["dataframes"]["Thrust"]
            elif "Deck" in enginedata["dataframes"]:
                df = enginedata["dataframes"]["Deck"]
            else:
                continue  # Skip the test if there is no appropriate dataframe

            if all(x in df for x in
                   ["Thrust [N]", "Mach Number", "Altitude [m]"]):
                M = df["Mach Number"].to_numpy()
                h = df["Altitude [m]"].to_numpy()
                T = df["Thrust [N]"].to_numpy()

                if enginedeck.type != "turboprop":
                    condition = np.isclose(
                        enginedeck.thrust(mach=M, altitude_m=h), T
                    ).all()
                else:
                    # The private method has no keyword arguments!
                    condition = np.isclose(getattr(
                        enginedeck, "_f_thrust_core")(M, h), T
                                           ).all()
            # Appropriate selection of columns to test could not be found
            else:
                continue

            self.assertTrue(condition)

        return


if __name__ == '__main__':
    unittest.main()
