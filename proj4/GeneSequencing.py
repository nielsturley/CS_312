#!/usr/bin/python3
import math

from cell import Cell

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import random

# Used to compute the bandwidth for banded version
MAXINDELS = 3

# Used to implement Needleman-Wunsch scoring
MATCH = -3
INDEL = 5
SUB = 1

class GeneSequencing:

    def __init__( self ):
        pass

    # This is the method called by the GUI.  _seq1_ and _seq2_ are two sequences to be aligned, _banded_ is a boolean that tells
    # you whether you should compute a banded alignment or full alignment, and _align_length_ tells you
    # how many base pairs to use in computing the alignment

    def align( self, seq1, seq2, banded, align_length):
        self.banded = banded
        self.MaxCharactersToAlign = align_length
        alignment1 = ''
        alignment2 = ''
        score = 0

        # If banded, compute the bandwidth
        if banded:
            score, alignment1, alignment2 = self.compute_banded_alignment(seq1, seq2)
            if score == math.inf:
                return{'align_cost':score, 'seqi_first100':alignment1, 'seqj_first100':alignment2}
        else:
            score, alignment1, alignment2 = self.compute_alignment(seq1, seq2)

        alignment1 = alignment1[:100]
        alignment2 = alignment2[:100]


        return {'align_cost':score, 'seqi_first100':alignment1, 'seqj_first100':alignment2}

    # Time complexity: O(n * m). n and m are the lengths of the sequences. We have to iterate through each cell once,
    # computing the best path/score each time.

    # Space complexity: O(n * m). n and m are the lengths of the sequences.
    # The matrix contains a cell for each possible edit distance.
    def compute_alignment(self, seq1, seq2):
        length_1 = min(len(seq1) + 1, self.MaxCharactersToAlign + 1)
        length_2 = min(len(seq2) + 1, self.MaxCharactersToAlign + 1)
        # Create a matrix of cells
        matrix = [[Cell(0, None, '', '') for i in range(length_2)] for j in range(length_1)]

        # Initialize the first row and column
        for i in range(length_1):
            matrix[i][0] = Cell(i * INDEL, (i - 1, 0), seq1[i - 1], '-')
        for j in range(length_2):
            matrix[0][j] = Cell(j * INDEL, (0, j - 1), '-', seq2[j - 1])

        # Fill in the rest of the matrix
        for i in range(1, length_1):
            for j in range(1, length_2):
                # Calculate the score for a match/mismatch
                if seq1[i - 1] == seq2[j - 1]:
                    match_score = MATCH
                else:
                    match_score = SUB
                match = matrix[i - 1][j - 1].score + match_score

                # Calculate the score for an insertion
                insert = matrix[i - 1][j].score + INDEL

                # Calculate the score for a deletion
                delete = matrix[i][j - 1].score + INDEL

                # Determine the best score. tie goes insert first, then delete, then match
                if insert <= match and insert <= delete:
                    matrix[i][j] = Cell(insert, (i - 1, j), seq1[i - 1], '-')
                elif delete <= match and delete <= insert:
                    matrix[i][j] = Cell(delete, (i, j - 1), '-', seq2[j - 1])
                else:
                    matrix[i][j] = Cell(match, (i - 1, j - 1), seq1[i - 1], seq2[j - 1])

        # Traceback to find the alignment
        alignment1 = ''
        alignment2 = ''
        i = length_1 - 1
        j = length_2 - 1
        while i > 0 or j > 0:
            alignment1 = matrix[i][j].char1 + alignment1
            alignment2 = matrix[i][j].char2 + alignment2
            i, j = matrix[i][j].parent

        return matrix[length_1 - 1][length_2 - 1].score, alignment1, alignment2

    # Time complexity: O(n * k). n is the length of the first sequence, and k is (MAXINDELS * 2 + 1). We iterate
    # through the larger n * m matrix, but only in a narrow, diagonal band of cells, where each cell is only
    # MAXINDELS (in this case, 3 indels) apart from any other cell. Each cell is visited once to calculate the best
    # score/path.

    # Space complexity: O(n * k). n is the length of the first sequence, and k is (MAXINDELS * 2 + 1). We only create
    # and store a matrix for the band of cells.
    def compute_banded_alignment(self, seq1, seq2):
        length_1 = min(len(seq1) + 1, self.MaxCharactersToAlign + 1)
        length_2 = min(len(seq2) + 1, self.MaxCharactersToAlign + 1)
        if abs(length_1 - length_2) > MAXINDELS:
            return math.inf, 'No Alignment Possible', 'No Alignment Possible'

        # Create a matrix of cells
        matrix = [[Cell(math.inf, None, '', '') for i in range(MAXINDELS * 2 + 1)] for j in range(length_1)]

        # Initialize the first row and column
        for i in range(MAXINDELS + 1):
            matrix[i][0] = Cell(i * INDEL, (i - 1, 0), seq1[i - 1], '-')
        for j in range(MAXINDELS + 1):
            matrix[0][j] = Cell(j * INDEL, (0, j - 1), '-', seq2[j - 1])

        # Fill in the rest of the matrix
        for i in range(1, length_1):
            for j in range(MAXINDELS * 2 + 1):
                start_or_end = False
                if i - MAXINDELS <= 0:
                    k = j
                    start_or_end = True
                    if k > i + MAXINDELS or j == 0:
                        continue
                elif length_1 - MAXINDELS <= i:
                    k = length_1 - (2 * MAXINDELS) + j - 1
                    start_or_end = True
                    if k < i - MAXINDELS:
                        continue
                else:
                    k = i - MAXINDELS + j

                # use different parent cell scheme for start_or_end
                if start_or_end:
                    # Calculate the score for a match/mismatch
                    if seq1[i - 1] == seq2[k - 1]:
                        match_score = MATCH
                    else:
                        match_score = SUB
                    match = matrix[i - 1][j - 1].score + match_score

                    # Calculate the score for an insertion
                    insert = matrix[i - 1][j].score + INDEL

                    # Calculate the score for a deletion
                    delete = matrix[i][j - 1].score + INDEL

                    # Determine the best score. tie goes insert first, then delete, then match
                    if insert <= match and insert <= delete:
                        matrix[i][j] = Cell(insert, (i - 1, j), seq1[i - 1], '-')
                    elif delete <= match and delete <= insert:
                        matrix[i][j] = Cell(delete, (i, j - 1), '-', seq2[k - 1])
                    else:
                        matrix[i][j] = Cell(match, (i - 1, j - 1), seq1[i - 1], seq2[k - 1])
                else:
                    # Calculate the score for a match/mismatch
                    if seq1[i - 1] == seq2[k - 1]:
                        match_score = MATCH
                    else:
                        match_score = SUB
                    match = matrix[i - 1][j].score + match_score

                    # Calculate the score for an insertion
                    if j == MAXINDELS * 2:
                        insert = math.inf
                    else:
                        insert = matrix[i - 1][j + 1].score + INDEL

                    # Calculate the score for a deletion
                    delete = matrix[i][j - 1].score + INDEL

                    # Determine the best score. tie goes insert first, then delete, then match
                    if insert <= match and insert <= delete:
                        matrix[i][j] = Cell(insert, (i - 1, j + 1), seq1[i - 1], '-')
                    elif delete <= match and delete <= insert:
                        matrix[i][j] = Cell(delete, (i, j - 1), '-', seq2[k - 1])
                    else:
                        matrix[i][j] = Cell(match, (i - 1, j), seq1[i - 1], seq2[k - 1])

        # Traceback to find the alignment
        alignment1 = ''
        alignment2 = ''
        i = length_1 - 1
        j = MAXINDELS * 2
        while i > 0 or j > 0:
            alignment1 = matrix[i][j].char1 + alignment1
            alignment2 = matrix[i][j].char2 + alignment2
            i, j = matrix[i][j].parent

        return matrix[length_1 - 1][MAXINDELS * 2].score, alignment1, alignment2




    def error_compute_banded_alignment(self, seq1, seq2):
        length_1 = min(len(seq1) + 1, self.MaxCharactersToAlign + 1)
        length_2 = min(len(seq2) + 1, self.MaxCharactersToAlign + 1)
        if abs(length_1 - length_2) > MAXINDELS:
            return math.inf, 'No Alignment Possible', 'No Alignment Possible'

        # Create a matrix of cells
        matrix = [[Cell(math.inf, None, '', '') for i in range(MAXINDELS * 2 + 1)] for j in range(length_1)]

        # Initialize the first 'row' and 'column'
        for i in range(MAXINDELS + 1):
            matrix[0][i + MAXINDELS] = Cell(i * INDEL, (0, i + MAXINDELS - 1), '-', seq2[i - 1])
        for j in range(MAXINDELS + 1):
            matrix[j][MAXINDELS - j] = Cell(j * INDEL, (j - 1, MAXINDELS - j - 1), seq1[j - 1], '-')

        for i in range(1, length_1):
            for j in range(MAXINDELS * 2 + 1):
                k = i - MAXINDELS + j
                if k <= 0 or k > len(seq2):
                    continue

                # Calculate the score for a match/mismatch
                if i - 1 < len(seq1):
                    if seq1[i - 1] == seq2[k - 1]:
                        match_score = MATCH
                    else:
                        match_score = SUB
                    match = matrix[i - 1][j].score + match_score
                else:
                    match = math.inf

                # Calculate the score for an insertion
                if j == MAXINDELS * 2:
                    insert = math.inf
                else:
                    insert = matrix[i - 1][j + 1].score + INDEL

                # Calculate the score for a deletion
                delete = matrix[i][j - 1].score + INDEL

                # Determine the best score. tie goes insert first, then delete, then match
                # if insert <= match and insert <= delete:
                #     matrix[i][j] = Cell(insert, (i - 1, j - 1), seq1[i - 1], '-')
                # elif delete <= match and delete <= insert:
                #     matrix[i][j] = Cell(delete, (i, j - 1), '-', seq2[k - 1])
                # else:
                #     matrix[i][j] = Cell(match, (i - 1, j), seq1[i - 1], seq2[k - 1])

                if insert <= match and insert <= delete:
                    if i - 1 < len(seq1):
                        matrix[i][j] = Cell(insert, (i - 1, j - 1), seq1[i - 1], '-')
                    else:
                        matrix[i][j] = Cell(insert, (i - 1, j - 1), '-', '-')
                elif delete <= match and delete <= insert:
                    if k - 1 < len(seq2):
                        matrix[i][j] = Cell(delete, (i, j - 1), '-', seq2[k - 1])
                    else:
                        matrix[i][j] = Cell(delete, (i, j - 1), '-', '-')
                else:
                    if i - 1 < len(seq1) and k - 1 < len(seq2):
                        matrix[i][j] = Cell(match, (i - 1, j), seq1[i - 1], seq2[k - 1])
                    else:
                        matrix[i][j] = Cell(match, (i - 1, j), '-', '-')

        # Traceback to find the alignment
        alignment1 = ''
        alignment2 = ''
        i = length_1 - 1
        j = MAXINDELS
        while i > 0 or j > MAXINDELS:
            alignment1 = matrix[i][j].char1 + alignment1
            alignment2 = matrix[i][j].char2 + alignment2
            i, j = matrix[i][j].parent

        return matrix[length_1 - 1][MAXINDELS].score, alignment1, alignment2
