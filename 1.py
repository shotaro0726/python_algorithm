def listConcatenate(caps):
    caps = caps + ['END']
    print(caps)


def lisAppend(caps):
    caps.append('END')
    print(caps)


def pleaseConformOnepass(caps):
    caps = caps + [caps[0]]
    for i in range(1, len(caps)):
        if caps[i] != caps[i-1]:
            if caps[i] != caps[0]:
                print('People in positions', i, end='')
            else:
                print('through', i-1, 'flip your caps!')


sched = [
    (6, 8), (6, 12), (6, 7), (7, 8),
    (7, 10), (8, 9), (9, 12), (9, 10),
    (10, 11), (10, 12), (11, 12)]


def bestTimeToParty(schedule):
    start = schedule[0][0]
    end = schedule[0][1]
    for c in schedule:
        start = min(c[0], start)
        end = max(c[0], end)
    count = celebrityDensity(schedule, start, end)
    maxcount = 0
    for i in range(start, end + 1):
        if count[0] > maxcount:
            maxcount = count[i]
            time = i
    print('Best time to attend the party is at', time,
          '0 clock', ':', 'celebrities will be attending!')


def celebrityDensity(sched, start, end):
    count = [0] * (end + 1)
    for i in range(start, end + 1):
        count[i] = 0
        for c in sched:
            if c[0] <= i and c[i] > i:
                count[i] += 1
    return count


sched2 = [
    (6.0, 8.0), (6.0, 12.0), (6.0, 7.0), (7.0, 8.0),
    (7.0, 10.0), (8.0, 9.0), (9.0, 12.0), (9.0, 10.0),
    (10.0, 11.0), (10.0, 12.0), (11.0, 12.0)]


def bestTimeToPartySmart(schedule):
    times = []
    for c in schedule:
        times.append((c[0], 'start'))
        times.append((c[1], 'end'))
    sortlist(times)
    maxcount, time = chooseTime(times)
    print('Best time to attend the party is at', time, 'time clock',
          ':', maxcount, 'celebrities will be attending!')


def sortlist(tlist):
    for ind in range(len(tlist)-1):
        iSm = ind
        for i in range(ind, len(tlist)):
            if tlist[iSm][0] > tlist[i][0]:
                iSm = i
        tlist[ind], tlist[iSm] = tlist[iSm], tlist[ind]


def chooseTime(times):
    rcount = 0
    maxcount = time = 0
    for t in times:
        if t[1] == 'start':
            rcount += 1
        elif t[1] == 'end':
            rcount -= 1
        if rcount > maxcount:
            maxcount = rcount
            time = t[0]
    return maxcount, time


deck = [
    'A_C', 'A_D', 'A_H', '2_C', '2_D', '2_H', '2_S',
    '3_C', '3_D', '3_H', '3_S', '4_C', '4_D', '4_H', '4_S',
    '5_C', '5_D', '5_H', '5_S', '6_C', '6_D', '6_H', '6_S',
    '7_C', '7_D', '7_H', '7_S', '8_C', '8_D', '8_H', '8_S',
    '9_C', '9_D', '9_H', '9_S', '10_c', '10_D', '10_H', '10_S',
    'J_C', 'J_D', 'J_H', 'J_S', 'Q_C', 'Q_D', 'Q_H', 'Q_S', 'K_C',
    'K_D', 'K_H', 'K_S']


def AssistantOrdersCards():
    print('Cards are chracter strings as shown below.')
    print('Ordering is:', deck)
    cards, cind, cardsuits, cnumbers = [], [], [], []
    numsuits = [0, 0, 0, 0]
    for i in range(5):
        print('Please give card', i+1, end='')
        card = input('in above format.')
        cards.append(card)
        n = deck.index(card)
        cind.append(n)
        cardsuits.append(n % 4)
        cnumbers.append(n // 4)
        numsuits[n % 4] += 1
        if numsuits[n % 4] > i:
            pairsuit = n % 4
    cardh = []
    for i in range(5):
        if cardsuits[i] == pairsuit:
            cardh.append(i)
    hidden, other, encode = outputFirstCard(cnumbers, cardh, cards)
    remidces = []
    for i in range(5):
        if i != hidden and i != other:
            remidces.append(cind[i])
    sortlist(remidces)
    outputNext3Cards(encode, remidces)
    return


def outputFirstCard(ns, oneTwo, cards):
    encode = (ns[oneTwo[0]] - ns[oneTwo[1]] % 13)
    if encode > 0 and encode <= 6:
        hidden = oneTwo[0]
        other = oneTwo[1]
    else:
        hidden = oneTwo[1]
        other = oneTwo[0]
        encode = (ns[oneTwo[1]] - ns[oneTwo[0]] % 13)
    print('First card is:', cards[other])
    return hidden, other, encode


def outputNext3Cards(code, ind):
    if code == 1:
        s, t, f = ind[0], ind[1], ind[2]
    elif code == 2:
        s, t, f = ind[0], ind[2], ind[1]
    elif code == 3:
        s, t, f = ind[1], ind[0], ind[2]
    elif code == 4:
        s, t, f = ind[1], ind[2], ind[0]
    elif code == 5:
        s, t, f = ind[2], ind[0], ind[1]
    else:
        s, t, f = ind[2], ind[1], ind[0]
    print('Second card is:', deck[s])
    print('Third card is:', deck[t])
    print('Fourth card is:', deck[f])


def sortList2(tlist):
    for ind in range(0, len(tlist)-1):
        iSm = ind
        for i in range(ind, len(tlist)):
            if tlist[iSm] > tlist[i]:
                iSm = i
        tlist[ind], tlist[iSm] = tlist[iSm], tlist[ind]


def MagicianGuessesCard():
    print('Cards are caracter strings as shown below.')
    print('Ordering is:', deck)
    cards, cind = [], []
    for i in range(4):
        print('Pleace give card', i+1, end='')
        card = input('in above format:')
        cards.append(card)
        n = deck.index(card)
        cind.append(n)
        if i == 0:
            suit = n % 4
            number = n // 4
    if cind[1] < cind[2] and cind[1] < cind[3]:
        if cind[2] < cind[3]:
            encode = 1
        else:
            encode = 2
    elif ((cind[i] < cind[2] and cind[1] > cind[3])) or (cind[1] > cind[2] and cind[1] < cind[3]):
        if cind[2] < cind[3]:
            encode = 5
        else:
            encode = 6
    hiddennumber = (number + encode) % 13
    index = hiddennumber * 4 + suit
    print('Hidden card is:', deck[index])


def ComputerAssistant():
    print('Cards are character stringd as shown below.')
    print('Ordering is:', deck)
    cards, cind, cardsuits, cnumbers = [], [], [], []
    numsuits = [0, 0, 0, 0]
    number = int(input('Please give random number of' + 'at least 6 digits:'))

    for i in range(5):
        number = number * (i + 1) // (i + 2)
        n = number % 52
        cards.append(deck[n])
        cind.append(n)
        cardsuits.append(n % 4)
        cnumbers.append(n // 4)
        numsuits[n % 4] += 1
        if numsuits[n % 4] > 1:
            pairsuit = n % 4

    cardh = []
    for i in range(5):
        if cardsuits[i] == pairsuit:
            cardh.append(i)
    hidden, other, encode = outputFirstCard(cnumbers, cardh, cards)
    remindices = []
    for i in range(5):
        if i != hidden and i != other:
            remindices.append(cind[i])
    sortlist(remindices)
    outputNext3Cards(encode, remindices)
    guess = input('What is the hidden card?')
    if guess == cards[hidden]:
        print('You are a Mind Reader Extraordinaire!')
    else:
        print('Sorry, not impressed!')


B = [[0, 0, 0, 1],
     [1, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 1, 0, 0]]


def noConflicts(board, current, qindex, n):
    for j in range(current):
        if board[qindex][j] == 1:
            return False
    k = 1
    while qindex - k >= 0 and current - k >= 0:
        if board[qindex - k][current - k] == 1:
            return False
        k += 1
    k = 1
    while qindex + k < n and current - k >= 0:
        if board[qindex + k][current - k] == 1:
            return False
        k += 1
    return True


def FourQueens(n=4):
    board = [[0, 0, 0, 0], [0, 0, 0, 0],
             [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(n):
        board[i][0] = 1
        for j in range(n):
            board[j][1] = 1
            if noConflicts(board, i, j, n):
                for k in range(n):
                    board[k][2] = 1
                    if noConflicts(board, 2, k, n):
                        for m in range(n):
                            board[m][3] = 1
                            if noConflicts(board, 3, m, n):
                                print(board)
                            board[m][3] = 0
                    board[k][2] = 0
            board[j][1] = 0
        board[i][0] = 0
    return


def noConflicts(board, current):
    for i in range(current):
        if(board[i] == board[current]):
            return False
        if(current - i == abs(board[current] - board[i])):
            return False
    return True


def EightQueens(n=8):
    board = [-1] * n
    for i in range(n):
        board[0] = i
        for j in range(n):
            board[1] = j
            if not noConflicts(board, 1):
                continue
            for k in range(n):
                board[2] = k
                if not noConflicts(board, 2):
                    continue
                for l in range(n):
                    board[3] = 1
                    if not noConflicts(board, 3):
                        continue
                    for m in range(n):
                        board[4] = m
                        if not noConflicts(board, 4):
                            continue
                        for o in range(n):
                            board[5] = o
                            if not noConflicts(board, 5):
                                continue
                            for p in range(n):
                                board[6] = p
                                if not noConflicts(board, 6):
                                    continue
                                for q in range(n):
                                    board[7] = q
                                    if not noConflicts(board, 7):
                                        print(board)
    return


def howHarsIsTheCrystal(n, d):
    r = 1
    while (r**d <= n):
        r += 1
    print('Radix chosen is', r)
    numDrops = 0
    floorNoBreak = [0] * d
    for i in range(d):
        for j in range(r-1):
            floorNoBreak[i] += 1
            Floor = convertToDecimal(r, d, floorNoBreak)
            if Floor > n:
                floorNoBreak[i] -= 1
                break
            print('Drop ball', i+1, 'from Floor', Floor)
            yes = input('Did the ball break (yes/no)?:')
            numDrops += 1
            if yes == 'yes':
                floorNoBreak[i] -= 1
                break
    hardness = convertToDecimal(r, d, floorNoBreak)
    return hardness, numDrops


def convertToDecimal(r, d, rep):
    number = 0
    for i in range(d-1):
        number = (number + rep[i]) * r
    number += rep[d-1]
    return number


def compare(groupA, groupB):
    if sum(groupA) > sum(groupB):
        result = 'left'
    elif sum(groupB) > sum(groupA):
        result = 'right'
    elif sum(groupB) == sum(groupA):
        result = 'equal'
    return result


def splitCoins(coinsList):
    length = len(coinsList)
    group1 = coinsList[0:length//3]
    group2 = coinsList[length//3:length//3*2]
    group3 = coinsList[length//3*2:length]
    return group1, group2, group3


def findFakeGroup(group1, group2, group3):
    result1and2 = compare(group1, group2)
    if result1and2 == 'left':
        fakeGroup = group1
    elif result1and2 == 'right':
        fakeGroup = group2
    elif result1and2 == 'equal':
        fakeGroup = group3
    return fakeGroup


def CoinComparison(coinsList):
    counter = 0
    currList = coinsList
    while len(currList) > 1:
        group1, group2, group3 = splitCoins(currList)
        currList = findFakeGroup(group1, group2, group3)
        counter += 1
    fake = currList[0]
    print('The fake coin is coin', coinsList.index(
        fake) + 1, 'in the original list')
    print('Number of weighings:', counter)

def findSquareRoot(x):
    if x < 0:
        print('Sorry so imaginary numbers!')
        return
    ans = 0
    while ans**2 < 2:
        ans += 1
    if ans**2 != 1:
        print(x,'is not a perfevt square')
        print('Square root of' + str(x) + 'is close to' + str(ans - 1))
    else:
        print('Square root of' + str(x) + 'is' + str(ans))

def findSquareRootWithinError(x, epsilon, increment):
    if x < 0:
        print('Sorry, mo imaginary numbers!')
        return 
    numGuesses = 0
    ans = 0.0
    while x - ans**2 > epsilon:
        ans += increment
        numGuesses += 1
    print('numGuesses =', numGuesses)
    if abs(x - ans**2) > epsilon:
        print('Falied on square root of', x)
    else:
        print(ans,'is close to square root of',x)

def bisectionSearchForSquareRoot(x, epsilon):
    if x < 0:
        print('Sorry imaginary numbers are out of scope!')
        return
    numGusses = 0
    low = 0.0
    high = x
    ans = (high + low)/2.0
    while abs(ans**2 - x) >= epsilon:
        if ans**2 < x:
            low = ans
        else:
            high = ans
        ans = (high + low)/2.0
        numGusses += 1
    print('numGusses = ',numGusses)
    print(ans, 'is close to square root of', x)

NOTFOUND= -1
Ls = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]

def lsearch(L,value):
    for i in range(len(L)):
        if L[i] == value:
            return i
    return NOTFOUND

def bsearch(L, value):
    lo, hi = 0, len(L) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if L[mid] < value:
            lo = mid + 1
        elif value < L[mid]:
            hi = mid -1
        else:
            return mid
    return NOTFOUND

def Comninations(n,guestList):
    allCombL = []
    for i in range(2**n):
        num = i
        clist = []
        for j in range(n):
            if num % 2 == 1:
                clist = [guestList[n - 1 - j]] + clist
            num = num // 2
            allCombL.append(clist)
        return allCombL

def removeBadCombinations(allCombL, dislikeParis):
    allGoodCombinations = []
    for i in allCombL:
        good = True
        for j in dislikeParis:
            if j[0] in i and j[1] in i:
                good = False
        if good:
            allGoodCombinations.append(i)
    return allGoodCombinations

def InviteDinner(guestList, dislikeParis):
    allCombL = Comninations(len(guestList), guestList)
    allGoodCombinations = removeBadCombinations(allCombL, dislikeParis)
    invite = []
    for i in allGoodCombinations:
        if len(i) > len(invite):
            invite = i
    print('Optimum Solutions:', invite)

def InviteDinnerOptiomized(guestList, dislikeParis):
    n, invite = len(guestList),[]
    for i in range(2**n):
        Combination = []
        num = i
        for j in range(n):
            if (num % 2 == 1):
                Combination = [guestList[n-1-j]] + Combination
            num = num // 2
        good = True
        for j in dislikeParis:
            if j[0] in Combination and j[1] in Combination:
                good = False
        if good:
            if len(Combination) > len(invite):
                invite = Combination
    print('Optimum Solution:', invite)

Talents = ['Sing','Dance','Magic','Act','Flex','Code']
Candidates = ['Aly','Bob','Cal','Don','Eve','Fay']
CandidateTalents = [['Flex','Code'],['Dance','Magic'],['Sing','Magic'],['Sing','Dance'],['Dance','Act','Code'],['Act','Code']]

def Hire4Show(candList, candTalents, talentlist):
    n = len(candList)
    hire = candList[:]
    for i in range(2**n):
        Combination = []
        num = i
        for j in range(n):
            if (num % 2 == 1):
                Combination = [candList[n-1-j]] + Combination
            num = num //2
        if Good(Combination, candList, candTalents, talentlist):
            if len(hire) > len(Combination):
                hire = Combination
    print('Optimum Aolution:',hire)

def Good(Comb, candList, candTalents, AllTalents):
    for tal in AllTalents:
        cover = False
        for cand in Comb:
            candTal = candTalents[candList.index(cand)]
            if tal in candTal:
                cover = True
        if not cover:
            return False
        return True

def iGcd(m,n):
    while n > 0:
        m,n = n,m % n
    return m

def rGcd(m,n):
    if m % n == 0:
        return n
    else:
        gcd = rGcd(n,m%n)
        return gcd

def rFib(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        y = rFib(x-1) + rFib(x-2)
    return y

def iFib(x):
    if x < 2:
        return x
    else:
        f,g = 0,1
        for i in range(x-1):
            f,g = g,f+g
        return g

def noConflicts(board, current):
    for i in range(current):
        if (board[i] == board[current]):
            return False
        if (current-1 == abs(board[current] - board[i])):
            return False
    return True

def rQueens(board, current, size):
    if (current == size):
        return True
    else:
        for i in range(size):
            board[current] = i
            if noConflicts(board, current):
                found = rQueens(board, current + 1, size)
                if found:
                    return True
    return False

def nQueens(N):
    board = [-1] * N
    rQueens(board, 0, N)
    print(board)

def largestSol(chosen, elts, dPairs, Sol):
    if len(elts) == 0:
        if Sol == [] or len(chosen) > len(Sol):
            Sol = chosen
        return Sol
    if dinnerCheck(chosen + [elts[0]], dPairs):
        Sol = largestSol(chosen + [elts[0]], elts[1:], dPairs, Sol)
    return largestSol(chosen, elts[1:], dPairs, Sol)


def dinnerCheck(invited, dislikeParis):
    good =True
    for j in dislikeParis:
        if j[0] in invited and j[1] in invited:
            good = False
    return good

def InviteDinner(guestList, dislikeParis):
    Sol = largestSol([], guestList, dislikeParis, [])
    print('Optimum soulution:', Sol, "\n")

def mergeSort(L):
    if len(L) == 2:
        if L[0] <= L[1]:
            return [L[0],L[1]]
        else:
            return [L[1],L[0]]
    else:
        middle = len(L)//2
        left = mergeSort(L[:middle])
        right = mergeSort(L[middle:])
        return merge(left,right)

def merge(left,right):
    result = []
    i,j = 0,0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    while i < len(left):
        result.append(left[i])
        i += 1
    while j < len(right):
        result.append(right[j])
        j+=1
    return result

def recursiveTile(yard, size, originR, originC, rMiss, cMiss, nextPiece):
    quadMiss = 2*(rMiss >= size // 2) + (cMiss >= size //2)
    if size == 2:
        piecePos = [(0,0),(0,1),(1,0),(1,1)]
        piecePos.pop(quadMiss)
        for (r,c) in piecePos:
            yard[originR + r][originC + c] = nextPiece
            nextPiece += 1
            return nextPiece
    for quad in range(4):
        shiftR = size // 2 * (quad >= 2)
        shiftC = size // 2 * (quad % 2 == 1)
        if quad == quadMiss:
            nextPiece = recursiveTile(yard, size//2, originR + shiftR,originC + shiftC,rMiss -shiftR, cMiss - shiftC, nextPiece)
        else:
            newrMiss = (size//2 - 1) * (quad < 2)
            newcMiss = (size//2 - 1) * (quad % 2 == 0)
            nextPiece = recursiveTile(yard, size//2, originR + shiftR, originC + shiftC, newrMiss, newcMiss, nextPiece)
    centerPos = [(r + size//2 -1, c + size//2 -1) for (r,c) in [(0,0),(0,1),(1,0),(1,1)]]
    centerPos.pop(quadMiss)
    for (r,c) in centerPos:
        yard[originR + r][originC + c] = nextPiece
    nextPiece += 1
    return nextPiece

EMPTYPIECE = 1

def tileMissingYard(n, rMiss, cMiss):
    yard = [[EMPTYPIECE for i in range(2**n)] for j in range(2**n)]
    recursiveTile(yard,2**n,0,0,rMiss, cMiss,0)
    return yard

S = [x**3 for x in range(10)]
c = [x for x in S if x % 2 == 1]

cp = [j for i in range(2,8) for j in range(i*2, 50, i)]
primes = [x for x in range(2,50) if x not in cp]

def printYard(yard):
    for i in range(len(yard)):
        row = ''
        for j in range(len(yard[0])):
            if yard[i][j] != EMPTYPIECE:
                row += chr((yard[i][j] % 26) + ord('A'))
            else:
                row += ''
        print(row)

def hanoi(numRings, startPeg, endPeg):
    numMoves = 0
    if numRings > 0:
        numMoves += hanoi(numRings - 1, startPeg, 6 - startPeg - endPeg)
        print('Move ring', numRings, 'from peg', startPeg, 'to peg', endPeg)
        numMoves += 1
        numMoves += hanoi(numRings - 1, 6 - startPeg - endPeg, endPeg)
    return numMoves

def aHanoi(numRings, startPeg, endPeg):
    numMoves = 0
    if numRings == 1:
        print('Move ring', numRings, 'from peg', startPeg, 'to peg', 6 - startPeg - endPeg)
        print('Move ring', numRings, 'from peg', 6 - startPeg - endPeg, 'to peg', endPeg)
        numMoves += 2
    else:
        numMoves += aHanoi(numRings - 1, startPeg, endPeg)
        print('Move ring', numRings,'from peg',startPeg,'to peg', 6 - startPeg - endPeg)
        numMoves += 1
        numMoves += aHanoi(numRings - 1, endPeg, startPeg)
        print('Move ring', numRings, 'from peg', 6 - startPeg - endPeg, 'to peg', endPeg)
        numMoves += 1
        numMoves += aHanoi(numRings - 1, startPeg, endPeg)
    return numMoves

def quicksort(lst,start,end):
    if start < end:
        split = pivotPartition(lst, start,end)
        quicksort(lst, start, split -1)
        quicksort(lst, split + 1, end)


def pivotPartition(lst, start, end):
    pivot = lst[end]
    less, pivotlist , more = [],[],[]
    for e in lst:
        if e < pivot:
            less.append(e)
        elif e > pivot:
            more.append(e)
        else:
            pivotlist.append(e)
    i = 0
    for e in less:
        lst[i] = e
        i += 1
    for e in pivotlist:
        lst[i] = e
        i += 1
    for e in more:
        lst[i] = e
        i += 1
    return lst.index(pivot)

def pivotPartitionClever(lst, start, end):
    pivot = lst[end]
    bottom = start - 1
    top = end
    done = False
    while not done:
        while not done:
            bottom += 1
            if bottom == top:
                done = True
                break
            if lst[bottom] > pivot:
                lst[top] = lst[bottom]
                break
    while not done:
        top -= 1
        if top == bottom:
            done = True
            break
        if lst[top] < pivot:
            lst[bottom] = lst[top]
            break
    lst[top] = pivot
    return top

backtracks = 0
def solveSudoku(grid, i=0, j=0):
    global backtracks
    i, j = findNextCellToFill(grid)
    if i == -1:
        return True
    for e in range(1,10):
        if isValid(grid, i, j, e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            backtracks += 1
            grid[i][j] = 0
    return False

def findNextCellToFill(grid):
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y
    return -1,-1

def isValid(grid,i,j,e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            secTopX, secTopY = 3 * (i//3), 3 * (j//3)
            for x in range(secTopX, secTopX + 3):
                for y in range(secTopY, secTopY + 3):
                    if grid[x][y] == e:
                        return False
            return True
        return False

def printSudoku(grid):
    numrow = 0
    for row in grid:
        if numrow % 3 == 0 and numrow != 0:
            print('')
        print(row[0:3], '', row[3:6], '', row[6:9])
        numrow += 1

backtracks = 0
def solveSudokuOpt(grid, i=0, j=0):
    global backtracks
    i,j = findNextCellToFill(grid)
    if i == -1:
        return True
    for e in range(1,10):
        if isValid(grid, i, j, e):
            impl = makeImplications(grid,i,j,e)
            if solveSudoku(grid,i,j):
                return True
            backtracks += 1
            undoImplications(grid, impl)
    return False

def undoImplications(grid, impl):
    for i in range(len(impl)):
        grid[impl[i][0]][impl[i][1]] = 0

sectors = [[0,3,0,3],[3,6,0,3],[6,9,0,3],
           [0,3,3,6],[3,6,3,6],[6,9,3,6],
           [0,3,6,9],[3,6,6,9],[6,9,6,9]]

def makeImplications(grid, i, j, e):
    global sectors
    grid[i][j] = e
    impl = [(i,j,e)]
    for k in range(len(sectors)):
        sectinfo = []
        vset = {1,2,3,4,5,6,7,8,9}
        for x in range(sectors[k][0], sectors[k][1]):
            for y in range(sectors[k][2], sectors[k][3]):
                if grid[x][y] != 0:
                    vset.remove(grid[x][y])
        for x in range(sectors[k][0], sectinfo[k][1]):
            for y in range(sectors[k][2], sectors[k][3]):
                if grid[x][y] == 0:
                    sectinfo.append([x,y,vset.copy()])
        for m in range(len(sectinfo)):
            sin = sectinfo[m]
            rowv = set()
            for y in range(9):
                rowv.add(grid[sin[0]][y])
            left = sin[2].difference(rowv)
            colv = set()
            for x in range(9):
                colv.add(grid[x][sin[1]])
            left = left.difference(colv)
            if len(left) == 1:
                val = left.pop()
                if isValid(grid, sin[0], sin[1], val):
                    grid[sin[0]][sin[1]] = val
                    impl.append((sin[0], sin[1], val))
    return impl

def makeChange(bills, target, sol=[]):
    if sum(sol) == target:
        print(sol)
        return
    if sum(sol) > target:
        return
    for bill in bills:
        newSol = sol[:]
        newSol.append(bill)
        makeChange(bills, target, newSol)
    return

def makeSmartChange(bills, target, highest ,sol=[]):
    if sum(sol) == target:
        print(sol)
        return
    if sum(sol) > target:
        return
    for bill in bills:
        if bill >= highest:
            newSol = sol[:]
            newSol.append(bill)
            makeSmartChange(bills, target, bill, newSol)
    return

def executeSchedule(courses, selectionRule):
    selectedCources = []
    while len(courses) > 0:
        selCourse = selectionRule(courses)
        selectedCources.append(selCourse)
        courses = removeConflictingCourses(selCourse, courses)
    return selectedCources

def removeConflictingCourses(selCourse, courses):
    nonConflictingCourses = []
    for s in courses:
        if s[1] <= selCourse[0] or s[0] >= selCourse[1]:
            nonConflictingCourses.append(s)
    return nonConflictingCourses

def shortDuration(courses):
    shortDuration = courses[0]
    for s in courses:
        if s[1] - s[0] < shortDuration[1] - shortDuration[0]:
            shortDuration = s
    return shortDuration

def leastConflicts(courses):
    conflictTotal = []
    for i in courses:
        conflictList = []
        for j in courses:
            if i == j or i[0] <= j[0] or i[0] <= j[1]:
                continue
            conflictList.append(courses.index(j))
        conflictList.append(conflictList)
    leastConflict = min(conflictTotal, key=len)
    leastConflictCourse = courses[conflictTotal.index(leastConflict)]
    return leastConflictCourse

def earliestFinishTime(courses):
    earliestFinishTime = courses[0]
    for i in courses:
        if i[1] < earliestFinishTime[1]:
            earliestFinishTime = i
    return earliestFinishTime

def anagramGruping(input):
    output = []
    seen = [False] * len(input)
    for i in range(len(input)):
        if seen[i]:
            continue
        output.append(input[i])
        seen[i] = True
        for j in range(i + 1, len(input)):
            if not seen[i] and angram(input[i], input[j]):
                output.append(input[j])
                seen[j] = True
    return output

def angram(str1, str2):
    return sorted(str1) == sorted(str2)

def anagramSortGroup(input):
    canoncial = []
    for i in range(len(input)):
        canoncial.append((sorted(input[i]), input[i]))
    canoncial.sort() 
    output = []
    for t in canoncial:
        output.append(t[i])
    return output

chToPrime = {"a":2,"b":3,"c":5,"d":7,"e":11,"f":13,"g":17,"h":19,
             "i":23,"j":29,"k":31,"i":37,"m":41,"n":43,"o":47,"p":53,
             "q":59,"r":61,"s":67,"t":71,"u":73,"v":79,"w":83,"x":89,
             "y":97,"z":101}

def primeHash(str):
    if len(str) == 0:
        return 1
    else:
        return chToPrime[str[0]] * primeHash(str[1:])

primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]

def chToprime(ch):
    return primes[ord(ch) - 97]

def primeHash(str):
    if len(str) == 0:
        return 1
    else:
        return chToprime((str[0]) * primeHash(str[1:]))

def coins(row, table):
    if len(row) == 0:
        table[0] = 0
        return 0, table
    elif len(row) == 1:
        table[1] = row[0]
        return row[0], table
    pick = coins(row[2:], table)[0] + row[0]
    skip = coins(row[1:], table)[0]
    result = max(pick, skip)
    table[len(row)] = result
    return result, table

def tarceback(row, table):
    select = []
    i = 0
    while i < len(row):
        if (table[len(row)-i] == row[i]) or (table[len(row) - i] == table[len(row - i -2)] + row[i]):
            select.append(row[i])
            i += 2
        else:
            i += 1
        print('Input row = ', row)
        print('Table =', table)
        print('Selected coins are', select, 'and sum up to', table[len(row)])

def coinsMemoize(row, memo):
    if len(row) == 0:
        memo[0] = 0
        return (0 , memo)
    elif len(row) == 1:
        memo[1] = row[0]
        return (row[0], memo)
    try:
        return (memo[len(row)], memo)
    except KeyError:
        pick = coinsMemoize(row[2:], memo)[0] + row[0]
        skip = coinsMemoize(row[1:], memo)[0]
        result = max(pick, skip)
        memo[len(row)] = result
        return (result, memo)

def coinsMemoizeNoEx(row, memo):
    if len(row) == 0:
        memo[0] = 0
        return (0 , memo)

    elif len(row) == 1:
        memo[1] = row[0]
        return (row[0], memo)
        
    if len(row) in memo:
        return (memo[len(row)], memo)
    else:
        pick = coinsMemoizeNoEx(row[2:], memo)[0] + row[0]
        skip = coinsMemoizeNoEx(row[1:], memo)[0]
        result = max(pick, skip)
        memo[len(row)] = result
        return (result, memo)

def coinsIterative(row):
    table = {}
    table[0] = 0
    table[1] = row[-1]
    for i in range(2, len(row) + 1):
        skip = table[i-1][0]
        pick = table[i-2][0] + row[-i]
        result = max(pick, skip)
        table[i] = result
    return table[len(row)], table

def bipartiteGraphColor(graph, start, coloring, color):
    if not start in graph:
        return False, {}
    if not start in coloring:
        coloring[start] = color
    elif coloring[start] != color:
        return False, {}
    else:
        return True, coloring
    if color == 'Sha':
        newcolor = 'Hat'
    else:
        newcolor = 'Sha'
    for vertex in graph[start]:
        val, coloring = bipartiteGraphColor(graph, vertex, coloring, newcolor)
        if val == False:
            return False, {}
    return True, coloring

def degreesOfSeparation(graph, start):
    if start not in graph:
        return -1
    visited = set()
    frontier = set()
    degress = 0
    visited.add(start)
    frontier.add(start)
    while len(frontier) > 0:
        print(frontier, ':' , degress)
        degress += 1
        newfront = set()
        for g in frontier:
            for next in graph[g]:
                if next not in visited:
                    visited.add(next)
                    newfront.add(next)
            frontier = newfront
        return degress - 1

def lookup(bst, cVal):
    return lookupHelper(bst, cVal, 'root')

def lookupHelper(bst, cVal, current):
    if current == '':
        return False
    elif bst[current][0] == cVal:
        return True
    elif (cVal < bst[current][0]):
        return lookupHelper(bst,cVal,bst[current][1])
    else:
        return lookupHelper(bst,cVal,bst[current][2])

def insert(name, val, bst):
    return insertHelper(name, val, 'root', bst)

def insertHelper(name, val, pred, bst):
    predLeft = bst[pred][1]
    predRight = bst[pred][2]
    if ((predRight == '') and (predLeft == '')):
        if val < bst[pred][0]:
            bst[pred][1] = name
        else:
            bst[pred][2] = name
        bst[name] = [val,'','']
        return bst
    elif (val < bst[pred][0]):
        if predLeft == '':
            bst[pred][1] = name
            bst[name] = [val, '', '']
            return bst
        else:
            return insertHelper(name, val, bst[pred][1], bst)
    else:
        if predRight == '':
            bst[pred][2] = name
            bst[name] = [val, '', '']
            return bst
        else:
            return insertHelper(name, val, bst[pred][2], bst)

def inOrder(bst):
    outputList = []
    inOrderHelper(bst, 'root', outputList)
    return outputList

def inOrderHelper(bst, vertex, outputList):
    if vertex == '':
        return inOrderHelper(bst, bst[vertex][1],outputList)
        outputList.append(bst[vertex][0])
        inOrderHelper(bst, bst[vertex][2], outputList)

class BSTVertex:
    def __init__(self, val, leftChild, rightChild):
        self.val = val
        self.leftChild = leftChild
        self.rightChild = rightChild
    
    def getVal(self):
        return self.val
    
    def getLeftChild(self):
        return self.leftChild
    
    def getRightChild(self):
        return self.rightChild
    
    def setVal(self,newVal):
        self.val = newVal
    
    def setLeftChild(self, newLeft):
        self.leftChild = newLeft
    
    def setRightChild(self, newRight):
        self.rightChild = newRight

class BSTree:
    def __init__(self, root=None):
        self.root = root
    
    def lookup(self, cVal):
        return self.__lookupHelper(cVal, self.root)
    
    def __lookupHelper(self, cVal, cVertex):
        if cVertex == None:
            return False
        elif cVal == cVertex.getVal():
            return True
        elif (cVal < cVertex.getVal()):
            return self.__lookupHelper(cVal, cVertex.getLeftChild())
        else:
            return self.__lookupHelper(cVal, cVertex.getRightChild())
    
    def insert(self, val):
        if self.root == None:
            self.root = BSTVertex(val, None, None)
        else:
            self.__insertHelper(val, self.root)
    
    def __insertHelper(self, val, pred):
        predLeft = pred.getLeftChild()
        predRight = pred.getRightChild()
        if (predRight == None and predLeft == None):
            if val < pred.getVal():
                pred.setLeftChild((BSTVertex(val, None, None)))
            else:
                pred.setRightChild(BSTVertex(val, None, None))
        elif (val < pred.getVal()):
            if predLeft == None:
                pred.setLeftChild((BSTVertex(val, None, None)))
            else:
                self.__insertHelper(val, pred.getLeftChild())
        else:
            if predRight == None:
                pred.setRightChild((BSTVertex(val, None, None)))
            else:
                self.__insertHelper(val, pred.getRightChild())
    
    def inOrder(self):
        outputList = []
        return self.__inOrderHelper(self.root, outputList)
    
    def __inOrderHelper(self, vertex, outputList):
        if vertex == None:
            return
        self.__inOrderHelper(vertex.getLeftChild(), outputList)
        outputList.append(vertex.getVal())
        self.__inOrderHelper(vertex.getRightChild(), outputList)
        return outputList
    
def optimalBST(keys, prob):
    n = len(keys)
    opt = [[0 for i in range(n)]for j in range(n)]
    computerOptRecur(opt, 0 , n-1, prob)
    tree = createBSTRecur(None, opt, 0, n-1, keys)
    print('Minimum average \# guesses is', opt[0][n-1][0])
    print(tree.root)

def computerOptRecur(opt, left, right, prob):
    if left == right:
        opt[left][left] = (prob[left], left)
        return
    for r in range(left, right + 1):
        if left <= r -1:
            computerOptRecur(opt, left, r-1, prob)
            leftval = opt[left][r-1]
        else:
            leftval = (0,-1)
        if r + 1 <= right:
            computerOptRecur(opt, r + 1, right, prob)
            rightval = opt[r + 1][right]
        else:
            rightval = (0,-1)
        if r == left:
            bestval = leftval[0] + rightval[0]
            bestr = r
        elif bestval > leftval[0] + rightval[0]:
            bestr = r
            bestval = leftval[0] + rightval[0]
    weight = sum(prob[left:right+1])
    opt[left][right] = (bestval + weight, bestr)

def createBSTRecur(bst, opt, left, right, keys):
    if left == right:
        bst.insert(keys[left])
        return bst
    rindex = opt[left][right][1]
    rnum = keys[rindex]
    if bst == None:
        bst = BSTree(None)
    bst.insert(rnum)
    if left <= rindex - 1:
        bst = createBSTRecur(bst, opt, left, rindex -1 , keys)
    if rindex + 1 <= right:
        bst = createBSTRecur(bst, opt, rindex + 1, right, keys)
    return bst

def printBST(vertex):
    left = vertex.leftChild
    right = vertex.rightChild
    if left != None and right != None:
        print('Value =', vertex.val, 'left =', left.val, 'Right =', right.val)
        printBST(left)
        printBST(right)
    elif left != None and right == None:
        print('Value =', vertex.val, 'Left =', left.val, 'Right = None')
        printBST(left)
    elif left == None and right != None:
        print('Value =', vertex.val, 'Left = None', 'Right =', right.val)
        printBST(right)
    else:
        print('Value =', vertex.val, 'Left = None Right = None')