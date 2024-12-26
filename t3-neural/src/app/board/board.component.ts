import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BoardService } from '../board.service';

@Component({
  selector: 'app-board',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './board.component.html',
  styleUrl: './board.component.scss',
  providers: [BoardService]
})
export class BoardComponent {
  board: number[] = new Array(9).fill(0);
  visited: boolean[] = new Array(9).fill(false);
  boardStates: number[][] = [];
  player: number = 1;
  curr: number = 1;

  wins: number = 0;
  losses: number = 0;
  draws: number = 0;

  constructor(private boardService: BoardService) { }

  mark(idx: number): void {
    if (this.visited[idx]) return;
    this.board[idx] = this.curr;
    this.visited[idx] = true;
    this.curr = this.curr === 1 ? -1 : 1;
    this.boardStates.push([...this.board]);
    if (!this.checkWin()) {
      this.playbot();
    }
  }

  checkWin(): boolean {
    let win_criterion = [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
      [0, 3, 6],
      [1, 4, 7],
      [2, 5, 8],
      [0, 4, 8],
      [2, 4, 6],
    ];
    for (let i = 0; i < 8; i++) {
      if (this.board[win_criterion[i][0]] !== 0) {
        if (
          this.board[win_criterion[i][0]] === this.board[win_criterion[i][1]] &&
          this.board[win_criterion[i][0]] === this.board[win_criterion[i][2]]
        ) {
          this.visited = this.visited.fill(true);
          if (this.curr === -1) {
            this.wins++;
            this.boardStates = [];
          }
          else {
            this.losses++;
            this.boardStates = [];
          }
          return true;
        }
      }
    }
    if (this.visited.filter((x) => x === true).length === 9) {
      this.draws++;
      this.boardStates = [];
      return true;
    }
    return false;
  }

  resetBoard(): void {
    this.board = this.board.fill(0);
    this.visited = this.visited.fill(false);
    this.curr = 1;
  }

  playbot() {
    this.boardService
      .ai_move({ team: this.curr == 1 ? 'X' : 'O', board: this.board })
      .subscribe((data: any) => {
        let idx = data.move;
        this.board[idx] = this.curr;
        this.visited[idx] = true;
        this.curr = this.curr === 1 ? -1 : 1;
        this.boardStates.push([...this.board]);
        this.checkWin();
      });
  }
}
