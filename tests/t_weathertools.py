"""Unit test for the weathertools module."""
import unittest

from ADRpy.weathertools import decodemetar


class DecodeMetar(unittest.TestCase):

    @staticmethod
    def test_decodemetar():
        metar_str = "METAR EGHI 181450Z 22010KT 180V260 9999 BKN014 13/10 Q1028"
        _ = decodemetar(metar_str)
        return


if __name__ == '__main__':
    unittest.main()
