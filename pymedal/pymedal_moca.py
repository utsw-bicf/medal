################################################################################
## Extension of Medal algorithm Described in: Patient similarity in the longitudinal care  ##
## of patients with polypharmacy treatment (Pineda, et. al.)                  ##
##                                                                            ##
## Implemented by Armin Pourshafeie                                           ##
##                                                                            ##
##                                                                            ##
################################################################################


import numpy as np
from numba import  jit,  njit, prange
import numba as nb
import pandas as pd
import time
import tqdm
import sys
import os


START= 1
END  = -1
CONT = 2
DASH = -2


def _getSequence(pat):
    """
    Encodes a patient usage data into sequence vectors.
    Relies on START, CONT, END values defined globally.

    Args:
        pat: Usage dataframe for a single patient.

    Returns:
        List of unique test.
        List of sequences (list of numpy arrays).
        List of start visit.
        List of end visit.
    """
    moca = np.unique(pat.moca)
    sequences = []
    startVisits = []
    endVisits = []
    for mocat_test in moca:
        tested_by = pat.loc[pat.moca==mocat_test, :]
        endVisit = np.max(tested_by.visit)
        startVisit = np.min(tested_by.visit)
        startVisits.append(startVisit)
        endVisits.append(endVisit)
        seq = np.zeros(( endVisit - startVisit +1), dtype=np.int8)
        base = startVisit
        l = len(seq) - 2
        for i_ind, score in zip(tested_by.visit, tested_by.score):
            if not np.isnan(score):
                i = i_ind - base
                j = max(i, l)
                seq[i] = score
                seq[(i+1):j+1] = score
        sequences.append(seq)
    return(moca, sequences, startVisits, endVisits)

def getSequences(data):
    """
    Encodes the usage data into sequence vectors.

    Args:
        data: Dataframe with all patient's usage data.

    Returns:
        List with tuples of test, sequence, start visit and end visit for
        each patient
    """

    patients = sorted(np.unique(data.id))
    #npatients = len(patients)
    patientInfo = []
    for patient in patients:
        patientInfo.append(_getSequence(data.loc[data.id ==patient,:]))

    return patientInfo

@njit(nb.int32[:,:](nb.int8[:], nb.int8[:], nb.int32[:,:]), nogil=True,  parallel=False, cache=True)
def paths(seq1, seq2, mat):
    """
    Uses the medal algorithm to populate a cost matrix for different paths.

    Args:
        seq1: Patient1's sequence.
        seq2: Patient2's sequence.
        mat : int32[:,:] array of size seq1 + 1 x seq1 + 1

    Returns:
        Matrix of cost of various alignments.
    """
    mat[0,:] = 0
    mat[:,0] = 0
    for i in prange(len(seq1)):
        s1 = seq1[i]
        for j in range(len(seq2)):
            s2 = seq2[j]
            #neighbors = [ mat[i,j+1], mat[i,j], mat[i+1,j]] # t, d, l
            if s1 == s2: # Same case
                # WORK AROUND due to numba issues with min(arr)
                val = min(mat[i,j+1], mat[i,j], mat[i+1,j])
            else: # Scenario Changes
                if (s1+s2) % 2 == 0 : # Opposing Scenarios
                    val = max(mat[i,j+1], mat[i,j], mat[i+1,j]) + 1
                elif s1 + s2 == 1 and s1*s2 == 0: # Gap cases
                    val = min(mat[i,j+1], mat[i,j], mat[i+1,j]) + 1
                else:
                    val = max(mat[i,j+1], mat[i,j], mat[i+1,j])
            mat[i+1, j+1] = val
    return (mat)


@jit(nopython=True, nogil=True, cache=True)
def align(seq1, seq2, mat):
    """
    Traceback to determine alignment.

    Args:
        seq1: Expanded usage sequence for patient 1.
        seq2: Expanded usage sequence for patient 2.
        mat : medal alignment path costs.

    Returns:
        alignment for sequences 1,2 and the cost of the alignment.

    """
    i, j = len(seq1) - 1, len(seq2) - 1
    boolseq1 = seq1 > 0
    boolseq2 = seq2 > 0
    dist   = 0
    align1 = []
    align2 = []
    while (i >= 0 and j >= 0):
        if(boolseq1[i] == boolseq2[j]):
            align1.append(seq1[i])
            align2.append(seq2[j])
            i -= 1
            j -= 1
            dist += np.absolute(seq1[i] - seq2[j]) # do the difference in scores, best case zero
        elif mat[i, j-1] < mat[i-1, j]:
            align1.append(DASH)
            dist += 1
            align2.append(seq2[j])
            j -= 1
        else:
            align1.append(seq1[i])
            align2.append(DASH)
            dist += 1
            i -= 1
    if i >= 0:
        dist += i+1
        #align1 += seq1[i::-1].tolist()  # append the leftovers
        align1 += [item for item in seq1[i::-1]]  # append the leftovers
        align2 += [DASH for k in range(i+1)]
    elif j >= 0:
        dist += j+1
        #align2 += seq2[j::-1].tolist()  # append the leftovers
        align2 += [item for item in seq2[j::-1]]  # append the leftovers
        align1 += [DASH for k in range(j+1)]
    align1.reverse()
    align2.reverse()
    dist /= 2
    return(align1, align2, dist)



def medalDistance(patientInfo):
    """
    Computes the insertion edit distance between every pair of individuals.

    Args:
        patientInfo: List of meds, sequences, start and end times as produced by getSequence

    Returns:
        Pairwise distance matrix.

    """

    npatients = len(patientInfo)
    total_count = (npatients * (npatients - 1))/(2)
    update_step = int(total_count/100)
    distMat = np.zeros((npatients, npatients))
    pbar = tqdm.tqdm(total = 100)
    counter = 0
    for i in range(npatients):
        meds1, sequences1, startTimes1, endTimes1 = patientInfo[i]
        for j in range(0, i):
            dist, size = 0, 0
            meds2, sequences2, startTimes2, endTimes2 = patientInfo[j]
            inMed2 = meds2.tolist()
            for medInd, med in enumerate(meds1):
                if med not in meds2:
                    days  = endTimes1[medInd] - startTimes1[medInd] + 1
                    dist += days * days
                    size += days
                else:  # It's in both
                    medInd2      = np.where(meds2 == med)[0][0]
                    startTime    = min(startTimes1[medInd], startTimes2[medInd2])
                    endTime      = max(endTimes1[medInd], endTimes2[medInd2])
                    diff         = endTime - startTime + 1
                    expandedSeq1 = np.zeros(diff, dtype=np.int8)
                    expandedSeq2 = np.zeros(diff, dtype=np.int8)
                    expandedSeq1[startTimes1[medInd]-startTime:endTimes1[medInd]-startTime + 1] = sequences1[medInd]
                    expandedSeq2[startTimes2[medInd2]-startTime:endTimes2[medInd2]-startTime + 1] = sequences2[medInd2]
                    if endTimes1[medInd] < endTime:
                        expandedSeq1[endTimes1[medInd] - startTime + 1] = END
                    elif endTimes2[medInd2] < endTime:
                        expandedSeq2[endTimes2[medInd2] - startTime + 1] = END
                    mat = np.empty((diff + 1, diff + 1), dtype=np.int32)
                    mat = paths(expandedSeq1, expandedSeq2, mat)
                    al1, al2, d = align(expandedSeq1, expandedSeq2, mat)
                    size       += len(al1)
                    dist       += d * len(al1)
                    inMed2.remove(med)
            for med in inMed2:
                medInd2 = np.where(meds2 == med)[0][0]
                days    = endTimes2[medInd2] - startTimes2[medInd2] + 1
                dist   += days * days
                size   += days
            distMat[i,j] = dist/float(size)
            counter += 1
            if counter == update_step:
                pbar.update(1)
                counter = 0
    distMat += distMat.T
    return distMat


def medal(usageFile, idFileName="patientID.txt", distMatName="distance_mat.txt"):
    """
    Performs the medal alignment algorithms.

    Args:
        usageFile  : Drug usage file.
        idFileName : File path for the patient ids.
        distMatName: File path for the pairwise distance matrix.

    Returns: None. Writes the patient ids and distance matrix from the medal algorithm.
    """

    t = time.time()
    data = events = pd.read_csv(usageFile)
    patientInfo = getSequences(data)
    tPreprocess = time.time() - t

    patients = np.unique(data.id)
    del data
    np.savetxt("patientID.txt", patients, fmt='%i')
    #npatients = len(patients)

    tAlign = time.time()
    distMat = medalDistance(patientInfo)
    tAlign = time.time() - tAlign
    np.savetxt("distance_mat.txt", distMat)
    print("Preprocessing time: {}".format( tPreprocess ))
    print("Alignment time: {}".format( tAlign ))
    print("Total time: {}".format( time.time()-t ))

if __name__=="__main__":
    args = sys.argv
    usage = """USAGE: python pymedal datalocation\n """
    if len(args) < 2 or len(args) > 2:
        print (usage)
        print ("""pymedal only takes a single argument""")
        sys.exit(1)
    dataFile = args[1]
    if not os.path.isfile(dataFile):
        print (usage)
        print ("{} is not a file".format(dataFile))
        sys.exit(1)

    medal(dataFile)
